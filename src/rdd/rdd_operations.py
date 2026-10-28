"""RDD programming demo (INT315): transformations, actions, partitions, lazy evaluation.

Run:  python src/rdd/rdd_operations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spark_common import get_spark, banner, MAT_CSV  # noqa: E402


def category(g3):
    """Performance category from the final grade G3 (scale 0-20)."""
    if g3 >= 15:
        return "Excellent"
    if g3 >= 10:
        return "Pass"
    return "At-Risk"


def parse_line(line):
    """CSV line -> dict of the fields used in this demo (values are quoted strings)."""
    f = [x.strip('"') for x in line.split(";")]
    return {"school": f[0], "sex": f[1], "age": int(f[2]), "studytime": int(f[13]),
            "failures": int(f[14]), "absences": int(f[29]),
            "G1": int(f[30]), "G2": int(f[31]), "G3": int(f[32])}


def main():
    spark = get_spark("RDD-Operations")
    sc = spark.sparkContext

    banner("1. Creating an RDD from a Python list")
    rdd_list = sc.parallelize([12, 7, 15, 9, 18, 4, 10, 15, 7], numSlices=3)
    print("Contents (take):", rdd_list.take(10))
    print("Number of partitions:", rdd_list.getNumPartitions())
    print("Data per partition (glom):", rdd_list.glom().collect())

    banner("2. Loading student records from CSV (textFile)")
    lines = sc.textFile(MAT_CSV, minPartitions=4)
    header = lines.first()                       # action
    data = lines.filter(lambda l: l != header)   # transformation (lazy)
    print("Header:", header[:70], "...")
    print("Record count:", data.count())
    print("Partitions:", data.getNumPartitions())
    print("Records per partition:", data.glom().map(len).collect())

    students = data.map(parse_line).cache()
    print("First record:", students.first())

    banner("3. Transformations: map / filter / flatMap / distinct / union")
    g3 = students.map(lambda s: s["G3"])
    print("map      -> first 10 G3 values:", g3.take(10))
    high = students.filter(lambda s: s["G3"] >= 15)
    print("filter   -> students with G3 >= 15:", high.count())
    words = sc.parallelize(["good student", "at risk student"]).flatMap(lambda t: t.split())
    print("flatMap  -> words:", words.collect())
    print("distinct -> distinct studytime values:",
          sorted(students.map(lambda s: s["studytime"]).distinct().collect()))
    girls = students.filter(lambda s: s["sex"] == "F").map(lambda s: s["G3"])
    boys = students.filter(lambda s: s["sex"] == "M").map(lambda s: s["G3"])
    print("union    -> F + M G3 count:", girls.union(boys).count(), "(should equal total)")

    banner("4. Key-value RDD: reduceByKey / groupByKey / sortByKey")
    pairs = students.map(lambda s: (s["school"], s["G3"]))
    sums = pairs.map(lambda kv: (kv[0], (kv[1], 1))) \
                .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    print("reduceByKey -> average G3 per school:",
          sums.mapValues(lambda t: round(t[0] / t[1], 2)).collect())
    grouped = students.map(lambda s: (s["studytime"], s["G3"])).groupByKey() \
                      .mapValues(lambda v: round(sum(v) / len(v), 2))
    print("groupByKey  -> average G3 per studytime (sorted by key):", grouped.sortByKey().collect())
    print("sortByKey desc -> top 3 studytime keys:", grouped.sortByKey(ascending=False).take(3))

    banner("5. Actions: collect / count / first / take / reduce / countByKey")
    print("count       :", g3.count())
    print("first       :", g3.first())
    print("take(5)     :", g3.take(5))
    print("reduce(max) :", g3.reduce(lambda a, b: a if a > b else b))
    print("reduce(sum) :", g3.reduce(lambda a, b: a + b))
    print("collect() on a SMALL aggregated RDD:", sorted(sums.keys().collect()))
    print("countByKey  -> students per sex:",
          dict(students.map(lambda s: (s["sex"], 1)).countByKey()))

    banner("6. Average G3 and students per performance category")
    total, n = g3.map(lambda x: (x, 1)).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    print(f"Average G3 = {total / n:.2f}  (Spark built-in mean: {g3.mean():.2f})")
    cats = dict(students.map(lambda s: (category(s["G3"]), 1)).countByKey())
    for k in ("Excellent", "Pass", "At-Risk"):
        print(f"  {k:<10}: {cats.get(k, 0)}")
    print("Category rule: G3 >= 15 Excellent, 10-14 Pass, < 10 At-Risk")

    banner("7. Concepts (viva summary)")
    print("""\
- Transformation : builds a new RDD lazily (map, filter, reduceByKey ...).
- Action         : triggers execution and returns a result (count, take, reduce ...).
- Lazy evaluation: Spark records a lineage (DAG) and runs it only at an action,
                   so it can pipeline steps and avoid unnecessary work.
- Partition      : a chunk of the RDD; one task processes one partition, and
                   partitions are what run in parallel across cores/executors.
- Distributed    : data is split into partitions, tasks run in parallel, and
                   wide operations (reduceByKey, groupByKey) shuffle data by key.
- reduceByKey combines values inside each partition BEFORE the shuffle, so it
  is preferred over groupByKey for aggregation.""")
    print("\nLineage of the category-count RDD (toDebugString):")
    lineage = students.map(lambda s: (category(s["G3"]), 1)).reduceByKey(lambda a, b: a + b)
    print(lineage.toDebugString().decode())
    spark.stop()


if __name__ == "__main__":
    main()
