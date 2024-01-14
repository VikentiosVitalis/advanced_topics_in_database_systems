from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, desc, to_timestamp
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Start Spark session with 4 executors
spark = SparkSession.builder \
    .appName("CrimeDataAnalysis") \
    .config("spark.executor.instances", "4") \
    .getOrCreate()

# Read the datasets from the HDFS service
file_path_2010_to_2019 = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'
file_path_2020_to_present = 'hdfs://master:54310/datasets/Crime_Data_from_2020_to_Present.csv'

# The dataframes contain the data specified from the csv files
crime_data_2010_to_2019 = spark.read.csv(file_path_2010_to_2019, header=True, inferSchema=True)
crime_data_2020_to_present = spark.read.csv(file_path_2020_to_present, header=True, inferSchema=True)

# Union of the two dataframes
crime_data = crime_data_2010_to_2019.union(crime_data_2020_to_present)

# Convert 'DATE OCC' to timestamp, filter out null values in 'DATE OCC', extract year and month
crime_data = crime_data.withColumn('DATE OCC', to_timestamp(col('DATE OCC'), 'MM/dd/yyyy hh:mm:ss a'))
crime_data = crime_data.filter(crime_data['DATE OCC'].isNotNull())
crime_data = crime_data.withColumn('Year', year('DATE OCC'))
crime_data = crime_data.withColumn('Month', month('DATE OCC'))

# Dataframe has the columns 'Year', 'Month' and 'Crime Count'
grouped_data = crime_data.groupBy('Year', 'Month').count().withColumnRenamed('count', 'Crime Count')

# Create a window specification - rows in a frame
windowSpec = Window.partitionBy('Year').orderBy(desc('Crime Count'))

# Window function to assign rank, filter rows with rank <=3, order the result by year and rank
top_months = grouped_data.withColumn('Rank', F.rank().over(windowSpec)) \
                         .filter(col('Rank') <= 3) \
                         .orderBy('Year', 'Rank')

# Display the result 
top_months.show(top_months.count(), truncate=False)

# Stop the Spark session
spark.stop()
