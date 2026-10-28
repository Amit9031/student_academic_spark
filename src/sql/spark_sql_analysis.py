"""Spark SQL and DataFrame demonstrations with an interactive command-line menu.

Run (menu):            python src/sql/spark_sql_analysis.py
List the queries:      python src/sql/spark_sql_analysis.py --list
Run one query:         python src/sql/spark_sql_analysis.py --run 15
Run one and export:    python src/sql/spark_sql_analysis.py --run 15 --export
Run all queries:       python src/sql/spark_sql_analysis.py --all
Custom SQL:            python src/sql/spark_sql_analysis.py --sql "SELECT COUNT(*) FROM students"
DataFrame API demo:    python src/sql/spark_sql_analysis.py --dataframe
Join demo:             python src/sql/spark_sql_analysis.py --joins
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_common import get_spark, read_students, banner, OUTPUT_DIR  # noqa: E402
from sql_queries import QUERIES, get_query  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.window import Window  # noqa: E402

CSV_DIR = OUTPUT_DIR / "csv"


def category_expr():
    return (F.when(F.col("G3") >= 15, "Excellent")
             .when(F.col("G3") >= 10, "Pass")
             .otherwise("At-Risk"))


def load_data(spark):
    """Read the CSV, add student_id, and register the SQL views."""
    raw = read_students(spark)
    # The dataset has no ID column, so create one (1..N) in file order.
    w = Window.orderBy(F.monotonically_increasing_id())
    students = raw.withColumn("student_id", F.row_number().over(w)) \
                  .select("student_id", *raw.columns)
    students.createOrReplaceTempView("students")

    # Derived DataFrame: performance category for students 1-300 only,
    # plus two ids (9001, 9002) that do not exist in `students`.
    # The gaps on both sides make LEFT / RIGHT / FULL joins show different results.
    perf = (students.where(F.col("student_id") <= 300)
            .select("student_id", category_expr().alias("category")))
    extra = spark.createDataFrame([(9001, "Excellent"), (9002, "At-Risk")],
                                  ["student_id", "category"])
    perf = perf.unionByName(extra).withColumn(
        "support_action",
        F.when(F.col("category") == "At-Risk", "Academic support")
         .when(F.col("category") == "Pass", "Monitor")
         .otherwise("Enrichment"))
    perf.createOrReplaceTempView("performance")

    # Column names as data, so LIKE can be used on them.
    spark.createDataFrame([(c,) for c in students.columns], ["column_name"]) \
         .createOrReplaceTempView("column_names")
    return students, perf


# ---------------------------------------------------------------- DataFrame API
def dataframe_demo(spark, students):
    banner("DataFrame API demonstration")
    print("--- show() ---")
    students.select("student_id", "school", "sex", "G1", "G2", "G3").show(5)
    print("--- printSchema() (first 12 lines) ---")
    for line in students._jdf.schema().treeString().splitlines()[:12]:
        print(line)
    print("   ... (33 original columns + student_id)")
    print("--- select() ---")
    students.select("school", "sex", "studytime", "G3").show(3)
    print("--- filter() : G3 >= 18 ---")
    students.filter(F.col("G3") >= 18).select("student_id", "G1", "G2", "G3").show(5)
    print("--- withColumn() : improvement = G3 - G1 and category ---")
    students.withColumn("improvement", F.col("G3") - F.col("G1")) \
            .withColumn("category", category_expr()) \
            .select("student_id", "G1", "G3", "improvement", "category").show(5)
    print("--- describe() ---")
    students.describe("age", "absences", "G1", "G2", "G3").show()
    print("--- Python column inspection: names starting with 'G' ---")
    print([c for c in students.columns if c.startswith("G")])
    print("Note: LIKE matches VALUES in a column. To search column NAMES use\n"
          "students.columns / printSchema() / SHOW COLUMNS, or (as queries 8-10 do)\n"
          "put the names into a small table and run LIKE on that table.")


# ------------------------------------------------------------------- join demo
def join_demo(spark, students, perf):
    banner("Join demonstration (key: student_id)")
    print(f"students rows    : {students.count()}")
    print(f"performance rows : {perf.count()}  (students 1-300 + ids 9001, 9002)\n")
    for how in ["inner", "left", "right", "full"]:
        n = students.join(perf, on="student_id", how=how).count()
        print(f"{how.upper():<6} JOIN rows: {n}")
    print("""
Why the counts differ:
  INNER : only ids present on BOTH sides            -> 300
  LEFT  : all students, NULL category for 301-395   -> 395
  RIGHT : all performance rows, NULL G3 for 9001/2  -> 302
  FULL  : everything from both sides                -> 397""")

    banner("Duplicate-column issue")
    dup = students.alias("s").join(perf.alias("p"), F.col("s.student_id") == F.col("p.student_id"))
    print("Joining with an expression keeps BOTH student_id columns:")
    print("  columns containing 'student_id':", [c for c in dup.columns if c == "student_id"])
    try:
        dup.select("student_id").show(1)
    except Exception as e:  # AnalysisException: ambiguous reference
        print("  select('student_id') fails ->", type(e).__name__)
    print("Fix 1: join with on='student_id' (single merged key column)")
    print("  columns:", [c for c in students.join(perf, on='student_id').columns if c == "student_id"])
    print("Fix 2: select explicit aliases, e.g. s.student_id")
    dup.select(F.col("s.student_id"), "G3", "category").show(3)


# ------------------------------------------------------------------ run helpers
def run_sql(spark, sql, title=None, rows=20, export_name=None):
    if title:
        print(f"\n>>> {title}")
    print("SQL:", sql)
    df = spark.sql(sql)
    df.show(rows, truncate=False)
    if export_name:
        path = export_csv(df, export_name)
        print("Exported to:", path)
    return df


def export_csv(df, name):
    """Write a single CSV file output/csv/<name>.csv (Spark writes a folder; we pick out the part file)."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CSV_DIR / f"_tmp_{name}"
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(tmp))
    part = glob.glob(str(tmp / "part-*.csv"))[0]
    target = CSV_DIR / f"{name}.csv"
    shutil.move(part, target)
    shutil.rmtree(tmp, ignore_errors=True)
    return target


def run_catalogue_query(spark, qid, export=False):
    q = get_query(qid)
    if q is None:
        print(f"No query with id {qid}. Use --list to see 1-{len(QUERIES)}.")
        return None
    _id, title, concept, sql = q
    print(f"\n[{_id}] {title}\nConcept: {concept}")
    return run_sql(spark, sql, export_name=f"query_{_id:02d}" if export else None)


def list_queries():
    banner("Available SQL demonstrations")
    for qid, title, concept, _ in QUERIES:
        print(f"  {qid:>2}. {title}  [{concept}]")


# ------------------------------------------------------------------ interactive
MENU = """
Commands:
  <number>   run that query (1-{n})        l  list queries
  e <number> run and export to CSV         a  run all queries
  c          type your own SQL             d  DataFrame API demo
  j          join demo                     t  show tables/views
  h          show this menu                q  quit
Views: students, performance, column_names"""


def interactive(spark, students, perf):
    banner("Student Performance - Spark SQL console")
    print(f"Spark UI (while this is running): http://localhost:4040")
    print(MENU.format(n=len(QUERIES)))
    last = None
    while True:
        try:
            cmd = input("\nsql> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not cmd:
            continue
        low = cmd.lower()
        try:
            if low in ("q", "quit", "exit"):
                break
            elif low in ("h", "help", "?"):
                print(MENU.format(n=len(QUERIES)))
            elif low == "l":
                list_queries()
            elif low == "a":
                for q in QUERIES:
                    last = run_catalogue_query(spark, q[0])
            elif low == "d":
                dataframe_demo(spark, students)
            elif low == "j":
                join_demo(spark, students, perf)
            elif low == "t":
                spark.sql("SHOW TABLES").show()
            elif low == "c":
                print("Enter SQL (single line; empty line cancels). Views: students, performance, column_names")
                sql = input("custom> ").strip().rstrip(";")
                if sql:
                    last = run_sql(spark, sql)
            elif low.startswith("e ") and low[2:].strip().isdigit():
                last = run_catalogue_query(spark, int(low[2:].strip()), export=True)
            elif low.isdigit():
                last = run_catalogue_query(spark, int(low))
            elif low.startswith("select") or low.startswith("show") or low.startswith("with") \
                    or low.startswith("describe"):
                last = run_sql(spark, cmd.rstrip(";"))   # allow typing SQL directly
            else:
                print("Unknown command. Type h for help.")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"[error] {type(e).__name__}: {msg}")
    print("Closing Spark session. Goodbye.")


def main():
    ap = argparse.ArgumentParser(description="Spark SQL demonstrations")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", type=int, metavar="N")
    ap.add_argument("--export", action="store_true", help="with --run: also write CSV")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sql", type=str)
    ap.add_argument("--dataframe", action="store_true")
    ap.add_argument("--joins", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_queries()
        return

    spark = get_spark("Spark-SQL-Analysis")
    try:
        students, perf = load_data(spark)
        if args.run:
            run_catalogue_query(spark, args.run, export=args.export)
        elif args.all:
            for q in QUERIES:
                run_catalogue_query(spark, q[0])
        elif args.sql:
            run_sql(spark, args.sql.rstrip(";"))
        elif args.dataframe:
            dataframe_demo(spark, students)
        elif args.joins:
            join_demo(spark, students, perf)
        else:
            interactive(spark, students, perf)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
