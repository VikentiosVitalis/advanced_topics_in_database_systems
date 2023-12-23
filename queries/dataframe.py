from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, DateType
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("CrimeDataAnalysis").getOrCreate()

file_path = 'hdfs://master:54310/datasets/Crime_Data_from_2010_to_2019.csv'

df = spark.read.csv(file_path, header=True, inferSchema=True)

df = df.withColumn("Date Rptd", col("Date Rptd").cast(DateType()))
df = df.withColumn("DATE OCC", col("DATE OCC").cast(DateType()))
df = df.withColumn("Vict Age", col("Vict Age").cast(IntegerType()))
df = df.withColumn("LAT", col("LAT").cast(DoubleType()))
df = df.withColumn("LON", col("LON").cast(DoubleType()))

df.printSchema()

print("Total Number of Rows:", df.count())

spark.stop()
