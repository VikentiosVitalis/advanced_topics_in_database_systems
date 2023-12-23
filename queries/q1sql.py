from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp

spark = SparkSession.builder \
    .appName("CrimeDataAnalysis") \
    .config("spark.executor.instances", "4") \
    .getOrCreate()

file_path_2010_to_2019 = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
file_path_2020_to_present = 'hdfs://master:54310/datasets/Crime_Data_from_2020_to_Present.csv'

crime_data_2010_to_2019 = spark.read.csv(file_path_2010_to_2019, header=True, inferSchema=True)
crime_data_2020_to_present = spark.read.csv(file_path_2020_to_present, header=True, inferSchema=True)

crime_data = crime_data_2010_to_2019.union(crime_data_2020_to_present)

crime_data = crime_data.withColumn('DATE OCC', to_timestamp('DATE OCC', 'MM/dd/yyyy hh:mm:ss a'))

crime_data.createOrReplaceTempView("crime_data")

query = """
SELECT Year, Month, `Crime Count`, Rank
FROM (
    SELECT 
        YEAR(`DATE OCC`) AS Year, 
        MONTH(`DATE OCC`) AS Month, 
        COUNT(*) AS `Crime Count`,
        RANK() OVER (PARTITION BY YEAR(`DATE OCC`) ORDER BY COUNT(*) DESC) AS Rank
    FROM crime_data
    WHERE `DATE OCC` IS NOT NULL
    GROUP BY Year, Month
) AS RankedData
WHERE Rank <= 3
ORDER BY Year, Rank
"""

top_months = spark.sql(query)

top_months.show(top_months.count(), truncate=False)

spark.stop()