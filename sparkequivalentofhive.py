#monthly_totals of withdrwal and deposits

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum, round as _round

# Initialize Spark session
spark = SparkSession.builder.appName("BankDataEDA").getOrCreate()

#Read the whole folder, NOT a specific CSV file
df = spark.read.option("header", True).csv("gs://bigassiiignment/processed/bank_data_for_eda_v2/", inferSchema=True)

# Check schema to confirm everything loaded correctly
df.printSchema()

#Task 1: Monthly Withdrawals and Deposits Summary
monthly_summary = df.groupBy("Year", "Month") \
    .agg(
        _round(_sum("Withdrawls"), 2).alias("Total_Withdrawals"),
        _round(_sum("Deposits"), 2).alias("Total_Deposits")
    ) \
    .orderBy("Year", "Month")

# Save result to GCS
monthly_summary.write.option("header", True).mode("overwrite").csv("gs://bigassiiignment/eda/statistics/monthly_totals_spark/")



#Transaction type frequency

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum as _sum, round as _round, desc

# Start session
spark = SparkSession.builder.appName("BankDataEDA").getOrCreate()

#Re-load the dataset (assuming the header is present and schema needs to be inferred)
df = spark.read.option("header", True).csv("gs://bigassiiignment/processed/bank_data_for_eda_v2/", inferSchema=True)

# Check schema for confirmation
df.printSchema()

txn_summary = df.groupBy("Txn_Type") \
    .agg(
        count("*").alias("Frequency"),
        _round(_sum("Withdrawls"), 2).alias("Total_Withdrawn"),
        _round(_sum("Deposits"), 2).alias("Total_Deposited")
    ) \
    .orderBy(desc("Frequency"))

txn_summary.write.option("header", True).mode("overwrite").csv("gs://bigassiiignment/eda/statistics/txn_type_summary_spark/")
