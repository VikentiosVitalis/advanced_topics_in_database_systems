from pyspark.sql import SparkSession
from pyspark import SparkContext

spark = SparkSession.builder \
    .appName("CrimeDataAnalysis") \
    .config("spark.executor.instances", "4") \
    .getOrCreate()

sc = spark.sparkContext

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

file_path_2010_to_2019 = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
file_path_2020_to_present = 'hdfs://master:54310/datasets/Crime_Data_from_2020_to_Present.csv'

crime_data_2010_to_2019 = sc.textFile(file_path_2010_to_2019)
crime_data_2020_to_present = sc.textFile(file_path_2020_to_present)

crime_data = crime_data_2010_to_2019.union(crime_data_2020_to_present)

header = crime_data.first()
crime_data = crime_data.filter(lambda line: line != header)

crime_data = crime_data.map(lambda line: line.split(','))\
                       .filter(lambda cols: 'STREET' in cols[15])\
                       .map(lambda cols: (classify_time_segment(int(cols[3])), 1))

segment_crime_counts = crime_data.reduceByKey(lambda a, b: a + b)

sorted_segment_crime_counts = segment_crime_counts.sortBy(lambda x: x[1], ascending=False)

for segment, count in sorted_segment_crime_counts.collect():
    print(f"{segment}: {count}")

spark.stop()
 