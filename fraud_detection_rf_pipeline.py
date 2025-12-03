
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Initialize Spark
spark = SparkSession.builder.appName("FraudDetectionRF").getOrCreate()

# Load data
data_path = "gs://bigassiiignment/processed/bank_data_for_eda_v2/BT_Processed.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(data_path)

# Feature columns and label
feature_cols = ["Withdrawls", "Deposits", "Balance", "Net_Change", "Balance_Drop",
                "Rolling_Withdrawals_3", "Txn_Count"]
label_col = "Large_Withdrawal_Flag"
df = df.withColumn(label_col, col(label_col).cast("int")).na.drop(subset=feature_cols + [label_col])

# Pipeline
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
rf = RandomForestClassifier(labelCol=label_col, featuresCol="features", numTrees=100)
pipeline = Pipeline(stages=[assembler, rf])

# Train/test split
train, test = df.randomSplit([0.7, 0.3], seed=42)
model = pipeline.fit(train)
predictions = model.transform(test)

# Evaluation
evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction")
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

# Confusion matrix
y_true = predictions.select(label_col).rdd.flatMap(lambda x: x).collect()
y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
cm = pd.crosstab(pd.Series(y_true, name='Actual'), pd.Series(y_pred, name='Predicted'))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.savefig("/tmp/rf_confusion_matrix.png")

# Save metrics
metrics = pd.DataFrame([{
    "Accuracy": accuracy,
    "F1 Score": f1,
    "Precision": precision,
    "Recall": recall
}])
metrics.to_csv("/tmp/rf_model_metrics.csv", index=False)

# Save feature importances
rf_model = model.stages[-1]
importances = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf_model.featureImportances.toArray()
})
importances.to_csv("/tmp/rf_feature_importance.csv", index=False)

# Upload to GCS
os.system("gsutil cp /tmp/rf_model_metrics.csv gs://bigassiiignment/model/evaluation/rf_model_metrics.csv")
os.system("gsutil cp /tmp/rf_confusion_matrix.png gs://bigassiiignment/model/visualizations/rf_confusion_matrix.png")
os.system("gsutil cp /tmp/rf_feature_importance.csv gs://bigassiiignment/model/visualizations/rf_feature_importance.csv")
