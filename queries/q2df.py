from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

spark = SparkSession.builder \
    .appName("CrimeDataAnalysis") \
    .config("spark.executor.instances", "4") \
    .getOrCreate()

def classify_time_segment(time):
    if 500 <= time < 1159:
        return 'Morning'
    elif 1200 <= time < 1659:
        return 'Afternoon'
    elif 1700 <= time < 2059:
        return 'Evening'
    elif (2100 <= time <= 2359) or (0 <= time < 459):
        return 'Night'
    else:
        return 'Undefined'

classify_time_segment_udf = udf(classify_time_segment, StringType())

file_path_2010_to_2019 = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
file_path_2020_to_present = 'hdfs://master:54310/datasets/Crime_Data_from_2020_to_Present.csv'

crime_data_2010_to_2019 = spark.read.csv(file_path_2010_to_2019, header=True, inferSchema=True)
crime_data_2020_to_present = spark.read.csv(file_path_2020_to_present, header=True, inferSchema=True)

crime_data = crime_data_2010_to_2019.union(crime_data_2020_to_present)

crime_data = crime_data.withColumn('Day Segment', classify_time_segment_udf(crime_data['TIME OCC']))

street_crimes = crime_data.filter(crime_data['Premis Desc'].like('%STREET%'))

segment_crime_counts = street_crimes.groupBy('Day Segment').count().withColumnRenamed('count', 'Crime Count')

sorted_segment_crime_counts = segment_crime_counts.orderBy('Crime Count', ascending=False)

sorted_segment_crime_counts.show()

spark.stop()
