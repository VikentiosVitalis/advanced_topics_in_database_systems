from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, year, udf, col, format_number, upper
from pyspark.sql.types import FloatType
import math

# Start Spark session 
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

# Convert 'DATE OCC' to date format and extract year
crime_data = crime_data.withColumn('DATE OCC', to_date('DATE OCC', 'MM/dd/yyyy hh:mm:ss a'))
crime_data = crime_data.withColumn('Year', year('DATE OCC'))

# Filter non null weapon crimes
weapon_crimes = crime_data.filter(crime_data['Weapon Used Cd'].isNotNull())

# Join dataframes with suffle replicate nl
weapon_crimes = weapon_crimes.hint("shuffle_replicate_nl")
weapon_crimes = weapon_crimes.join(police_stations, upper(weapon_crimes['AREA NAME']) == police_stations['DIVISION'], 'left_outer')

# Calculate distance
weapon_crimes = weapon_crimes.withColumn('Distance', get_distance_udf(weapon_crimes['LAT'], weapon_crimes['LON'], police_stations['Y'], police_stations['X']))

# Group, summ, rename and select columns
station_stats = weapon_crimes.groupBy('AREA NAME').agg(
    {'Distance': 'mean', 'DR_NO': 'count'}
).withColumnRenamed('avg(Distance)', 'Average_Distance')\
  .withColumnRenamed('count(DR_NO)', 'Count')\
  .withColumnRenamed('AREA NAME', 'Division')\
  .select('Division', format_number('Average_Distance', 3).alias('Average_Distance'), 'Count')\
  .orderBy('Count', ascending=False)

# Physical plan
station_stats.explain()

# Display the results
station_stats.show(station_stats.count(), truncate=False)

# Stop Spark session
spark.stop()