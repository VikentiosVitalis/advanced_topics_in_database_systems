from pyspark.sql import SparkSession
from pyspark import SparkContext

# Start Spark Session with 4 executors
spark = SparkSession.builder \
    .appName("CrimeDataAnalysis") \
    .config("spark.executor.instances", "4") \
    .getOrCreate()

# Retrieve Spark context associated with the current session 
sc = spark.sparkContext

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
    
# Read the datasets from the HDFS service
file_path_2010_to_2019 = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
file_path_2020_to_present = 'hdfs://master:54310/datasets/Crime_Data_from_2020_to_Present.csv'

# The dataframes contain the data specified from the text files
crime_data_2010_to_2019 = sc.textFile(file_path_2010_to_2019)
crime_data_2020_to_present = sc.textFile(file_path_2020_to_present)

# Union of the two dataframes
crime_data = crime_data_2010_to_2019.union(crime_data_2020_to_present)

# Extract header and filter it out
header = crime_data.first()
crime_data = crime_data.filter(lambda line: line != header)

# Data transformation
crime_data = crime_data.map(lambda line: line.split(','))\
                       .filter(lambda cols: 'STREET' in cols[15])\
                       .map(lambda cols: (classify_time_segment(int(cols[3])), 1))

# Reduce by key
segment_crime_counts = crime_data.reduceByKey(lambda a, b: a + b)

# Sort by count in descending order
sorted_segment_crime_counts = segment_crime_counts.sortBy(lambda x: x[1], ascending=False)

# Display results
for segment, count in sorted_segment_crime_counts.collect():
    print(f"{segment}: {count}")

# Stop Spark session
spark.stop()
 