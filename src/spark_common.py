"""Shared helpers: one place to build a SparkSession and locate project paths."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MAT_CSV = str(DATA_DIR / "student-mat.csv")
POR_CSV = str(DATA_DIR / "student-por.csv")

# Spark 4.2 / Scala 2.13 Kafka connector (used by the streaming unit)
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"


def get_spark(app_name="StudentPerformance", packages=None, shuffle_partitions=4):
    """Create a local SparkSession that uses the SAME Python as the running script
    (the venv), regardless of what ~/.zshrc exports."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.showConsoleProgress", "false")
    )
    if packages:
        builder = builder.config("spark.jars.packages", packages)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "ERROR"))
    quiet_query_error_logs()
    return spark


def quiet_query_error_logs():
    """PySpark 4.x logs a long JSON stack trace for every caught SQL/DataFrame error,
    which is noisy in a live demo. The exception is still raised normally.
    Set SPARK_VERBOSE_ERRORS=1 to see those logs when debugging."""
    if os.environ.get("SPARK_VERBOSE_ERRORS") == "1":
        return
    import logging
    for name in ("SQLQueryContextLogger", "DataFrameQueryContextLogger"):
        lg = logging.getLogger(name)
        lg.disabled = True
        lg.propagate = False


def read_students(spark, path=MAT_CSV):
    """Read a student CSV (semicolon separated)."""
    return spark.read.csv(path, header=True, sep=";", inferSchema=True)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
