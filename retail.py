from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, avg, desc

# Initializing Spark session
spark = SparkSession.builder.appName("RetailStoreAnalysis").getOrCreate()

# Correcingt file path in Google Cloud Storage (GCS)
file_path = "gs://aakanksha-bucket/retail-dataset.csv"  # Update with your correct GCS file path

# Loading the CSV file into a DataFrame
df = spark.read.option("header", "true").csv(file_path)

# Converting necessary columns to appropriate data types
df = df.withColumn("Salary", col("Salary").cast("float"))
df = df.withColumn("Age", col("Age").cast("int"))
df = df.withColumn("CustomerID", col("CustomerID").cast("int"))

# 1. Average Salary by Country
print("Average Salary by Country:")
df_grouped_by_country = df.groupBy("Country").agg(
    avg("Salary").alias("average_salary")
).orderBy(desc("average_salary"))
df_grouped_by_country.show()

# Saving the result of Average Salary by Country to GCS
df_grouped_by_country.write.csv("gs://aakanksha-bucket/output/average_salary_by_country", header=True)

# 2. Gender Distribution
print("\nGender Distribution:")
df_grouped_by_gender = df.groupBy("Gender").agg(
    count("CustomerID").alias("count")
).orderBy(desc("count"))
df_grouped_by_gender.show()

# Saving the result of Gender Distribution to GCS
df_grouped_by_gender.write.csv("gs://aakanksha-bucket/output/gender_distribution", header=True)

# 3.  calculating Average Age of Customers
print("\nAverage Age of Customers:")
average_age = df.agg(avg("Age").alias("average_age")).collect()
print(f"Average Age: {average_age[0]['average_age']}")

# Saving the result of Average Age to GCS (one record result, so we save it as a CSV)
average_age_df = spark.createDataFrame([(average_age[0]['average_age'],)], ["average_age"])
average_age_df.write.csv("gs://aakanksha-bucket/output/average_age.csv", header=True)

# 4.  calculating Total Customers by Country
print("\nTotal Customers by Country:")
df_grouped_by_country_customers = df.groupBy("Country").agg(
    count("CustomerID").alias("total_customers")
).orderBy(desc("total_customers"))
df_grouped_by_country_customers.show()

# Saving the result of Total Customers by Country to GCS
df_grouped_by_country_customers.write.csv("gs://aakanksha-bucket/output/total_customers_by_country", header=True)

# 5. printing Top 5 Countries by Total Salary Spend
print("\nTop 5 Countries by Total Salary Spend:")
df_grouped_by_country_salary = df.groupBy("Country").agg(
    sum("Salary").alias("total_salary_spent")
).orderBy(desc("total_salary_spent"))
df_grouped_by_country_salary.show(5)

# Saving the result of Top 5 Countries by Total Salary Spend to GCS
df_grouped_by_country_salary.write.csv("gs://aakanksha-bucket/output/top_5_countries_by_salary_spend", header=True)

# Stopping Spark session
spark.stop()
