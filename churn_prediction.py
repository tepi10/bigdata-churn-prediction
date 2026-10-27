
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Start Spark
spark = (
    SparkSession.builder
    .appName("CustomerChurnPrediction")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# Sample customer data:
# tenure_months, monthly_bill, support_calls, churn
data = [
    (2, 90.0, 5, 1),
    (4, 85.0, 4, 1),
    (6, 80.0, 4, 1),
    (8, 75.0, 3, 1),
    (10, 70.0, 3, 1),
    (12, 65.0, 2, 0),
    (18, 60.0, 2, 0),
    (24, 55.0, 1, 0),
    (30, 50.0, 1, 0),
    (36, 45.0, 0, 0),
    (3, 95.0, 6, 1),
    (9, 72.0, 4, 1),
    (15, 62.0, 2, 0),
    (20, 58.0, 1, 0),
    (28, 52.0, 0, 0),
    (5, 88.0, 5, 1),
    (14, 66.0, 2, 0),
    (32, 48.0, 1, 0),
    (7, 78.0, 3, 1),
    (26, 54.0, 1, 0)
]

columns = [
    "tenure_months",
    "monthly_bill",
    "support_calls",
    "churn"
]

df = spark.createDataFrame(data, columns)

print("Customer Dataset:")
df.show()

# Combine input columns into a feature vector
assembler = VectorAssembler(
    inputCols=["tenure_months", "monthly_bill", "support_calls"],
    outputCol="features"
)
prepared_df = assembler.transform(df).select("features", "churn")

# Split data into training and testing sets
train_df, test_df = prepared_df.randomSplit(
    [0.8, 0.2], seed=42
)

# Train logistic regression model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="churn",
    maxIter=20
)
model = lr.fit(train_df)

# Make predictions
predictions = model.transform(test_df)

print("Predictions:")
predictions.select("features", "churn", "prediction", "probability").show(
    truncate=False
)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(
    labelCol="churn",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator.evaluate(predictions)
print(f"Area Under ROC (AUC): {auc:.3f}")

spark.stop()
