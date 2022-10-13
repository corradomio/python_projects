# https://sparkbyexamples.com/pyspark/pyspark-read-csv-file-into-dataframe/

from pyspark.sql import SparkSession


spark: SparkSession = SparkSession.builder\
      .master("local[1]") \
      .appName("SparkByExamples.com")\
      .getOrCreate()

df = spark.read.option("header", True) \
    .csv("/Projects.github/python_projects/check_pyspark/allCountriesCSV.csv")
df.show(n=5)

df.filter(df.COUNTRY == "AD").show(truncate=False)

print(df.filter(df.COUNTRY == "AD").count())
print(df.filter(df.COUNTRY != "AD").count())
