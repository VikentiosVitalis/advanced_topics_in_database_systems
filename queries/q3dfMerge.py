from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, split, udf, regexp_replace, year
from pyspark.sql.types import StringType

# Start Spark session with 2 executors
spark = SparkSession.builder \
    .appName("CrimeVictimAnalysis") \
    .config("spark.executor.instances", "2") \
    .getOrCreate()

# Read the datasets from the HDFS service
crime_data_path = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
income_data_path = 'hdfs://master:54310/datasets/income/LA_income_2015.csv'
revgecoding_path = 'hdfs://master:54310/datasets/revgecoding.csv'

# Read data into Spark dataframes
crime_data = spark.read.csv(crime_data_path, header=True, inferSchema=True)
revgecoding = spark.read.csv(revgecoding_path, header=True, inferSchema=True)
income_data = spark.read.csv(income_data_path, header=True, inferSchema=True)

# Clean income data
income_data = income_data.withColumn('Estimated Median Income', regexp_replace('Estimated Median Income', '[\$,]', '').cast('float'))

# Transform the 'Estimated Median Income' column
crime_data = crime_data.withColumn('DATE OCC', to_date('DATE OCC', 'MM/dd/yyyy hh:mm:ss a'))

# Filter crime data for 2015
crime_2015 = crime_data.filter((year(col('DATE OCC')) == 2015) & (col('Vict Descent').isNotNull()))

# Join crime data with reverse geocoding data using merge join
crime_2015 = crime_2015.hint("merge").join(revgecoding, ['LAT', 'LON'], 'left_outer')
crime_2015 = crime_2015.withColumn('ZIPcode', split(col('ZIPcode'), ',').getItem(0))

# Extract first part of zip code, select top and bottom 3 zip codes by income
top_3_zip = income_data.orderBy('Estimated Median Income', ascending=False).limit(3)
bottom_3_zip = income_data.orderBy('Estimated Median Income', ascending=True).limit(3)
selected_zip_codes = top_3_zip.union(bottom_3_zip).select('Zip Code')

# Filter crimes for selected zip codes
selected_crimes = crime_2015.hint("merge").join(selected_zip_codes, crime_2015.ZIPcode == selected_zip_codes['Zip Code'])

# Define user defined function for descent mapping
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

# Apply udf to victim descent column
selected_crimes = selected_crimes.withColumn('Vict Descent', descent_udf('Vict Descent'))

# Group and count victims by descent
victim_count_by_descent = selected_crimes.groupBy('Vict Descent').count().orderBy('count', ascending=False)

# Physical Plan of execution plan
victim_count_by_descent.explain()

# Display results
victim_count_by_descent.show()

# Stop the Spark session
spark.stop()