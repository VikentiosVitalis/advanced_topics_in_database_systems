from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, desc, to_timestamp
from pyspark.sql.window import Window
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("CrimeDataAnalysis") \
    .config("spark.executor.instances", "4") \
    .getOrCreate()

file_path_2010_to_2019 = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
file_path_2020_to_present = 'hdfs://master:54310/datasets/Crime_Data_from_2020_to_Present.csv'

crime_data_2010_to_2019 = spark.read.csv(file_path_2010_to_2019, header=True, inferSchema=True)
crime_data_2020_to_present = spark.read.csv(file_path_2020_to_present, header=True, inferSchema=True)

crime_data = crime_data_2010_to_2019.union(crime_data_2020_to_present)


crime_data = crime_data.withColumn('DATE OCC', to_timestamp(col('DATE OCC'), 'MM/dd/yyyy hh:mm:ss a'))
crime_data = crime_data.filter(crime_data['DATE OCC'].isNotNull())
crime_data = crime_data.withColumn('Year', year('DATE OCC'))
crime_data = crime_data.withColumn('Month', month('DATE OCC'))

grouped_data = crime_data.groupBy('Year', 'Month').count().withColumnRenamed('count', 'Crime Count')

windowSpec = Window.partitionBy('Year').orderBy(desc('Crime Count'))

top_months = grouped_data.withColumn('Rank', F.rank().over(windowSpec)) \
                         .filter(col('Rank') <= 3) \
                         .orderBy('Year', 'Rank')

top_months.show(top_months.count(), truncate=False)

spark.stop()
