from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, split, udf, regexp_replace, year
from pyspark.sql.types import StringType

spark = SparkSession.builder \
    .appName("CrimeVictimAnalysis") \
    .config("spark.executor.instances", "2") \
    .getOrCreate()

crime_data_path = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
income_data_path = 'hdfs://master:54310/datasets/income/LA_income_2015.csv'
revgecoding_path = 'hdfs://master:54310/datasets/revgecoding.csv'

crime_data = spark.read.csv(crime_data_path, header=True, inferSchema=True)
revgecoding = spark.read.csv(revgecoding_path, header=True, inferSchema=True)
income_data = spark.read.csv(income_data_path, header=True, inferSchema=True)

income_data = income_data.withColumn('Estimated Median Income', regexp_replace('Estimated Median Income', '[\$,]', '').cast('float'))

crime_data = crime_data.withColumn('DATE OCC', to_date('DATE OCC', 'MM/dd/yyyy hh:mm:ss a'))

crime_2015 = crime_data.filter((year(col('DATE OCC')) == 2015) & (col('Vict Descent').isNotNull()))

crime_2015 = crime_2015.hint("shuffle_replicate_nl").join(revgecoding, ['LAT', 'LON'], 'left_outer')
crime_2015 = crime_2015.withColumn('ZIPcode', split(col('ZIPcode'), ',').getItem(0))

top_3_zip = income_data.orderBy('Estimated Median Income', ascending=False).limit(3)
bottom_3_zip = income_data.orderBy('Estimated Median Income', ascending=True).limit(3)
selected_zip_codes = top_3_zip.union(bottom_3_zip).select('Zip Code')

selected_crimes = crime_2015.hint("shuffle_replicate_nl").join(selected_zip_codes, crime_2015.ZIPcode == selected_zip_codes['Zip Code'])

selected_crimes.explain(True)

def descent_mapping(code):
    mapping = {
        'A': 'Other Asian', 'B': 'Black', 'C': 'Chinese', 'D': 'Cambodian',
        'F': 'Filipino', 'G': 'Guamanian', 'H': 'Hispanic/Latin/Mexican',
        'I': 'American Indian/Alaskan Native', 'J': 'Japanese', 'K': 'Korean',
        'L': 'Laotian', 'O': 'Other', 'P': 'Pacific Islander', 'S': 'Samoan',
        'U': 'Hawaiian', 'V': 'Vietnamese', 'W': 'White', 'X': 'Unknown',
        'Z': 'Asian Indian'
    }
    return mapping.get(code, 'Unknown')

descent_udf = udf(descent_mapping, StringType())

selected_crimes = selected_crimes.withColumn('Vict Descent', descent_udf('Vict Descent'))

victim_count_by_descent = selected_crimes.groupBy('Vict Descent').count().orderBy('count', ascending=False)

victim_count_by_descent.explain()

victim_count_by_descent.show()

spark.stop()