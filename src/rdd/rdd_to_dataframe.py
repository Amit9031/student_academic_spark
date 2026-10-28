"""Convert an RDD into a DataFrame using an EXPLICIT schema.

Run:  python src/rdd/rdd_to_dataframe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spark_common import get_spark, banner, MAT_CSV, OUTPUT_DIR  # noqa: E402
from pyspark.sql import Row  # noqa: E402
from pyspark.sql.types import StructType, StructField, StringType, IntegerType  # noqa: E402


def main():
    spark = get_spark("RDD-to-DataFrame")
    sc = spark.sparkContext

    lines = sc.textFile(MAT_CSV, minPartitions=4)
    header = lines.first()
    rows = lines.filter(lambda l: l != header).map(lambda l: [x.strip('"') for x in l.split(";")])

    banner("RDD of tuples -> DataFrame with an explicit schema")
    schema = StructType([
        StructField("school", StringType(), False),
        StructField("sex", StringType(), False),
        StructField("age", IntegerType(), False),
        StructField("studytime", IntegerType(), False),
        StructField("failures", IntegerType(), False),
        StructField("absences", IntegerType(), False),
        StructField("G1", IntegerType(), False),
        StructField("G2", IntegerType(), False),
        StructField("G3", IntegerType(), False),
    ])
    typed = rows.map(lambda f: (f[0], f[1], int(f[2]), int(f[13]), int(f[14]),
                                int(f[29]), int(f[30]), int(f[31]), int(f[32])))
    df = spark.createDataFrame(typed, schema)
    df.printSchema()
    df.show(5)
    print("Row count:", df.count())

    banner("Alternative: RDD of Row objects (schema inferred from Row fields)")
    row_rdd = typed.map(lambda t: Row(school=t[0], sex=t[1], G3=t[8]))
    spark.createDataFrame(row_rdd).show(3)

    banner("DataFrame -> RDD and a quick aggregation")
    print("df.rdd.take(2):", df.rdd.take(2))
    df.groupBy("school").avg("G3").show()

    out = str(OUTPUT_DIR / "csv" / "rdd_dataframe_sample")
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out)
    print("Saved DataFrame to:", out)
    spark.stop()


if __name__ == "__main__":
    main()
