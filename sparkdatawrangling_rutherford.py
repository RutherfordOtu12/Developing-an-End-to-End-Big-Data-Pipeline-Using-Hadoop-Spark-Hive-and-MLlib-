from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType, StringType, DateType

# Initialize Spark
spark = SparkSession.builder.appName("BankDataEDA").getOrCreate()

# Load the CSV
df = spark.read.option("header", True).csv("gs://bigassiiignment/Dataset/BT Records.csv")

# Remove duplicates
df = df.dropDuplicates()

# Drop rows missing essential fields
df = df.dropna(subset=["Date", "Balance"])

# Fill missing monetary values with 0
df = df.fillna({"Deposits": "0.00", "Withdrawls": "0.00"})

# Format numbers: remove commas, convert to Double
df = df.withColumn("Deposits", regexp_replace("Deposits", ",", "").cast(DoubleType())) \
       .withColumn("Withdrawls", regexp_replace("Withdrawls", ",", "").cast(DoubleType())) \
       .withColumn("Balance", regexp_replace("Balance", ",", "").cast(DoubleType()))

# Format date
df = df.withColumn("Date", to_date("Date", "dd-MMM-yyyy"))

# Clean and format text
df = df.withColumn("Description", trim(initcap(col("Description"))))

# Derive features
df = df.withColumn("Net_Change", col("Deposits") - col("Withdrawls")) \
       .withColumn("Day", dayofmonth("Date")) \
       .withColumn("Month", month("Date")) \
       .withColumn("Year", year("Date")) \
       .withColumn("Weekday", date_format("Date", "E")) \
       .withColumn("Week_Num", weekofyear("Date"))

# Sort by date for time series features
df = df.orderBy("Date")

# Window functions
from pyspark.sql.window import Window

window_spec = Window.orderBy("Date")

# Lag balance to detect large drops
df = df.withColumn("Prev_Balance", lag("Balance", 1).over(window_spec)) \
       .withColumn("Balance_Drop", col("Prev_Balance") - col("Balance"))

# Rolling sum of withdrawals over last 3 days
window_3 = Window.orderBy("Date").rowsBetween(-2, 0)
df = df.withColumn("Rolling_Withdrawals_3", sum("Withdrawls").over(window_3))

# Transaction frequency per day
df_txn_count = df.groupBy("Date").count().withColumnRenamed("count", "Txn_Count")
df = df.join(df_txn_count, on="Date", how="left")

# Flag large withdrawals (e.g., > 10,000)
df = df.withColumn("Large_Withdrawal_Flag", when(col("Withdrawls") > 10000, 1).otherwise(0))

# Categorize transaction type from description
df = df.withColumn("Txn_Type", 
                   when(col("Description").like("%Debit%"), "Card")
                   .when(col("Description").like("%Cash%"), "Cash")
                   .when(col("Description").like("%Interest%"), "Interest")
                   .when(col("Description").like("%Reversal%"), "Reversal")
                   .when(col("Description").like("%Commission%"), "Fee")
                   .otherwise("Other"))

# Save processed data
df.write.option("header", True).mode("overwrite") \
  .csv("gs://bigassiiignment/processed/bank_data_for_eda_v2")
