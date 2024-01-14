from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Start Spark session with 4 executors
spark = SparkSession.builder \
    .appName("CrimeDataAnalysis") \
    .config("spark.executor.instances", "4") \
    .getOrCreate()

# Devide the hours of a day to four segments
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

# User Defined Function to make it usable for dataFrame transformations
classify_time_segment_udf = udf(classify_time_segment, StringType())

# Read the datasets from the HDFS service
file_path_2010_to_2019 = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
file_path_2020_to_present = 'hdfs://master:54310/datasets/Crime_Data_from_2020_to_Present.csv'

# The dataframes contain the data specified from the csv files
crime_data_2010_to_2019 = spark.read.csv(file_path_2010_to_2019, header=True, inferSchema=True)
crime_data_2020_to_present = spark.read.csv(file_path_2020_to_present, header=True, inferSchema=True)

# Union of the two dataframes
crime_data = crime_data_2010_to_2019.union(crime_data_2020_to_present)

# Assign day segments based on time
crime_data = crime_data.withColumn('Day Segment', classify_time_segment_udf(crime_data['TIME OCC']))

# Filter street crimes
street_crimes = crime_data.filter(crime_data['Premis Desc'].like('%STREET%'))

# Group and count street crimes by day segment
segment_crime_counts = street_crimes.groupBy('Day Segment').count().withColumnRenamed('count', 'Crime Count')

# Sort the result by crime count in descending order
sorted_segment_crime_counts = segment_crime_counts.orderBy('Crime Count', ascending=False)

# Display reuslt
sorted_segment_crime_counts.show()

# Stop Spark session
spark.stop()
