from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, DateType
from pyspark.sql.functions import col

# Start Spark session
spark = SparkSession.builder.appName("CrimeDataAnalysis").getOrCreate()

# Read the datasets from the HDFS service
file_path = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'

# Dataframe with csv the columns and data types
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Data type casting in the new Dataframe
df = df.withColumn("Date Rptd", col("Date Rptd").cast(DateType()))
df = df.withColumn("DATE OCC", col("DATE OCC").cast(DateType()))
df = df.withColumn("Vict Age", col("Vict Age").cast(IntegerType()))
df = df.withColumn("LAT", col("LAT").cast(DoubleType()))
df = df.withColumn("LON", col("LON").cast(DoubleType()))

# Print the datatypes of the dataframe
df.printSchema()

# Print the number of rows in the dataframe
print("Total Number of Rows:", df.count())

# Stop the Spark Session 
spark.stop()
