from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, year, udf, col, format_number
from pyspark.sql.types import FloatType
import pandas as pd
import math

spark = SparkSession.builder \
    .appName("CrimeAnalysis") \
    .getOrCreate()

crime_data_path_2010_2019 = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
crime_data_path_2020_present = 'hdfs://master:54310/datasets/Crime_Data_from_2020_to_Present.csv'
police_stations_path = 'hdfs://master:54310/datasets/LAPD_Police_Stations.csv'

crime_data_2010_2019 = spark.read.csv(crime_data_path_2010_2019, header=True, inferSchema=True)
crime_data_2020_present = spark.read.csv(crime_data_path_2020_present, header=True, inferSchema=True)
police_stations = spark.read.csv(police_stations_path, header=True, inferSchema=True)

crime_data = crime_data_2010_2019.union(crime_data_2020_present)

crime_data = crime_data.withColumn('DATE OCC', to_date('DATE OCC', 'MM/dd/yyyy hh:mm:ss a'))
crime_data = crime_data.withColumn('Year', year('DATE OCC'))

firearm_crimes = crime_data.filter(crime_data['Weapon Used Cd'].between(100, 199))

police_stations_pd = police_stations.toPandas()
area_to_coords = {row['PREC']: (row['Y'], row['X']) for index, row in police_stations_pd.iterrows()}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

def get_distance(lat1, lon1, area):
    lat2, lon2 = area_to_coords.get(area, (0, 0))
    return haversine(lat1, lon1, lat2, lon2)

get_distance_udf = udf(get_distance, FloatType())

firearm_crimes = firearm_crimes.withColumn('Distance', get_distance_udf('LAT', 'LON', 'AREA '))

annual_stats = firearm_crimes.groupBy('Year').agg(
    {'Distance': 'mean', 'DR_NO': 'count'}
).select(
    "Year", format_number("avg(Distance)", 3).alias('Average_Distance'), "count(DR_NO)"
).withColumnRenamed('count(DR_NO)', 'Count')

annual_stats = annual_stats.orderBy('Year')

annual_stats.show()

spark.stop()

