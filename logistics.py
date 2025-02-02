from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initializing the Spark session
spark = SparkSession.builder \
    .appName("RetailStoreClassification") \
    .getOrCreate()

# Loading the dataset into a Spark DataFrame
file_path = "gs://aakanksha-bucket/retail-dataset.csv"
spark_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Dropping  rows with missing values
spark_df = spark_df.dropna()

# Encodeing categorical columns using StringIndexer
country_indexer = StringIndexer(inputCol="Country", outputCol="CountryIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="Label")

# Applying the indexers
spark_df = country_indexer.fit(spark_df).transform(spark_df)
spark_df = gender_indexer.fit(spark_df).transform(spark_df)

# Combining features into a single vector
feature_columns = ["Age", "Salary", "CountryIndex"]
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="Features")
spark_df = vector_assembler.transform(spark_df)

# Scaling the features
scaler = StandardScaler(inputCol="Features", outputCol="ScaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(spark_df)
spark_df = scaler_model.transform(spark_df)

# Splitting the data into training and testing sets
train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)

# Training a logistic regression model
lr = LogisticRegression(featuresCol="ScaledFeatures", labelCol="Label")
lr_model = lr.fit(train_data)

# Making predictions on the test set
predictions = lr_model.transform(test_data)

# Evaluating the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="Label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy:.2f}")

# Showing some predictions
predictions.select("Age", "Salary", "Country", "Gender", "prediction").show(10)

# Stopping the Spark session
spark.stop()
