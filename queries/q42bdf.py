from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, udf, col, format_number, initcap
from pyspark.sql.types import FloatType
import math

# Start Spark Session 
spark = SparkSession.builder \
    .appName("CrimeAnalysis") \
    .getOrCreate()

# Read the datasets from the HDFS service
crime_data_path_2010_2019 = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
crime_data_path_2020_present = 'hdfs://master:54310/datasets/Crime_Data_from_2020_to_Present.csv'
police_stations_path = 'hdfs://master:54310/datasets/LAPD_Police_Stations.csv'

# Read data into Spark dataframes
crime_data_2010_2019 = spark.read.csv(crime_data_path_2010_2019, header=True, inferSchema=True)
crime_data_2020_present = spark.read.csv(crime_data_path_2020_present, header=True, inferSchema=True)
police_stations = spark.read.csv(police_stations_path, header=True, inferSchema=True)

# Union of the the dataframes to one
crime_data = crime_data_2010_2019.union(crime_data_2020_present)

# Haversine function for the calculation of distance between 
# two sets of latitude and longitude coordinates
def haversine(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return None
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance if not math.isnan(distance) else None

get_distance_udf = udf(haversine, FloatType())

# Convert date format, filter weapon crimes
crime_data = crime_data.withColumn('DATE OCC', to_date('DATE OCC', 'MM/dd/yyyy hh:mm:ss a'))
weapon_crimes = crime_data.filter(crime_data['Weapon Used Cd'].isNotNull())

# Collect local police stations
local_police_stations = police_stations.collect()

# Finding nearest station function
def find_nearest_station(crime):
    min_distance = float('inf')
    nearest_station = None
    for station in local_police_stations:
        distance = haversine(crime['LAT'], crime['LON'], station['Y'], station['X'])
        if distance is not None and distance < min_distance:
            min_distance = distance
            nearest_station = station['DIVISION']
    return (crime['DR_NO'], nearest_station, min_distance)

# Mapping nearest stations RDD
nearest_stations_rdd = weapon_crimes.rdd.map(lambda crime: find_nearest_station(crime))

# Create dataframe from RDD
columns = ['DR_NO', 'Division', 'Distance']
nearest_stations_df = nearest_stations_rdd.toDF(columns)

# Group, sum nearest stations dataframe, column renaming
station_stats = nearest_stations_df.groupBy('Division').agg(
    {'Distance': 'mean', 'DR_NO': 'count'}
).withColumnRenamed('avg(Distance)', 'Average_Distance')\
  .withColumnRenamed('count(DR_NO)', 'Count')

# Data cleaning, formatting and sorting
station_stats = station_stats.withColumn('Division', initcap('Division'))
station_stats = station_stats.select(
    'Division', format_number('Average_Distance', 3).alias('Average_Distance'), 'Count'
).orderBy('Count', ascending=False)

# Display results
station_stats.show(station_stats.count(), truncate=False)

# Stop Spark Session
spark.stop()