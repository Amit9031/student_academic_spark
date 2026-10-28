"""Simple ML: predict student performance with Spark MLlib.

  1. Linear Regression  -> predict the final mark G3 (a number 0-20)
  2. Logistic Regression -> predict FAIL (G3 < 10) vs PASS

Each model is run in two scenarios:
  EARLY : uses only background features (no G1, G2)   -> prediction at the start of the course
  GRADES: also uses G1 and G2 (earlier marks)         -> prediction after two assessments
G3 is NEVER used as an input feature (that would be target leakage).

Run:  python src/ml/simple_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spark_common import get_spark, read_students, banner, OUTPUT_DIR  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler  # noqa: E402
from pyspark.ml.regression import LinearRegression  # noqa: E402
from pyspark.ml.classification import LogisticRegression  # noqa: E402
from pyspark.ml.evaluation import RegressionEvaluator  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

SEED = 42
NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
           "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
CATEGORICAL = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
               "guardian", "schoolsup", "famsup", "paid", "activities", "nursery",
               "higher", "internet", "romantic"]
SCENARIOS = {"EARLY": NUMERIC, "GRADES": NUMERIC + ["G1", "G2"]}
PASS_MARK = 10   # G3 < 10 is treated as FAIL (the "needs support" class)


def build_stages(numeric_cols, estimator):
    """StringIndexer -> OneHotEncoder -> VectorAssembler -> estimator.
    Numeric columns come first in the vector so coefficients can be labelled."""
    idx_out = [c + "_idx" for c in CATEGORICAL]
    ohe_out = [c + "_ohe" for c in CATEGORICAL]
    return [
        StringIndexer(inputCols=CATEGORICAL, outputCols=idx_out, handleInvalid="keep"),
        OneHotEncoder(inputCols=idx_out, outputCols=ohe_out, handleInvalid="keep"),
        VectorAssembler(inputCols=numeric_cols + ohe_out, outputCol="features"),
        estimator,
    ]


def run_linear(train, test, name, numeric_cols, results):
    banner(f"LINEAR REGRESSION - scenario {name}")
    print("Task    : regression (predict a number)")
    print("Target  : G3 (final maths mark, 0-20)")
    print(f"Features: {len(numeric_cols)} numeric + {len(CATEGORICAL)} categorical (one-hot encoded)")
    print("          G1/G2 included:", "G1" in numeric_cols)

    lr = LinearRegression(featuresCol="features", labelCol="G3", regParam=0.1, maxIter=100)
    model = Pipeline(stages=build_stages(numeric_cols, lr)).fit(train)   # fitted on TRAIN only
    pred = model.transform(test)

    lrm = model.stages[-1]
    print(f"\nIntercept: {lrm.intercept:.3f}")
    print("Coefficients of the numeric features:")
    for col, coef in zip(numeric_cols, lrm.coefficients.toArray()[:len(numeric_cols)]):
        print(f"  {col:<11}{coef:+.3f}")

    ev = lambda m: RegressionEvaluator(labelCol="G3", predictionCol="prediction", metricName=m).evaluate(pred)
    r2, rmse, mae = ev("r2"), ev("rmse"), ev("mae")
    train_mean = train.agg(F.avg("G3")).first()[0]
    base_rmse = test.select(F.sqrt(F.avg((F.col("G3") - F.lit(train_mean)) ** 2))).first()[0]
    print(f"\nTest rows : {test.count()}")
    print(f"R2   = {r2:.3f}   (1.0 is perfect, 0.0 is no better than predicting the average)")
    print(f"RMSE = {rmse:.3f}   (typical error in marks)")
    print(f"MAE  = {mae:.3f}   (average absolute error in marks)")
    print(f"Baseline RMSE if we always predict the training average: {base_rmse:.3f}")
    print("\nSample actual vs predicted:")
    pred.select("G3", F.round("prediction", 1).alias("predicted_G3")).show(8)
    results[f"linear_{name}"] = {"r2": r2, "rmse": rmse, "mae": mae, "baseline_rmse": base_rmse}


def run_logistic(train, test, name, numeric_cols, results):
    banner(f"LOGISTIC REGRESSION (pass/fail) - scenario {name}")
    print(f"Task    : binary classification  (label 1 = FAIL, G3 < {PASS_MARK};  label 0 = PASS)")
    print("Target  : fail_label (made from G3; G3 itself is not a feature)")
    print("Positive class = FAIL, because the goal is to find students who need support.")
    train = train.withColumn("label", (F.col("G3") < PASS_MARK).cast("double"))
    test = test.withColumn("label", (F.col("G3") < PASS_MARK).cast("double"))

    lg = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.1)
    model = Pipeline(stages=build_stages(numeric_cols, lg)).fit(train)
    pred = model.transform(test).cache()

    c = {(r["label"], r["prediction"]): r["count"]
         for r in pred.groupBy("label", "prediction").count().collect()}
    tp, fn = c.get((1.0, 1.0), 0), c.get((1.0, 0.0), 0)
    fp, tn = c.get((0.0, 1.0), 0), c.get((0.0, 0.0), 0)
    total = tp + tn + fp + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print(f"\nTest rows: {total}")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                    predicted PASS   predicted FAIL")
    print(f"  actual PASS (0)   TN = {tn:<10} FP = {fp}")
    print(f"  actual FAIL (1)   FN = {fn:<10} TP = {tp}")
    print("  TP: failing student correctly flagged     FP: passing student wrongly flagged")
    print("  TN: passing student correctly left alone  FN: failing student missed")
    print(f"\nAccuracy  = {acc:.3f}   (share of all predictions that are right)")
    print(f"Precision = {prec:.3f}   (of students flagged FAIL, how many really fail)")
    print(f"Recall    = {rec:.3f}   (of students who really fail, how many we caught)")
    print(f"F1-score  = {f1:.3f}   (balance of precision and recall)")
    print("\nSample predictions:")
    pred.select("G3", F.col("label").cast("int").alias("actual_fail"),
                F.col("prediction").cast("int").alias("predicted_fail")).show(8)
    results[f"logistic_{name}"] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                                   "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    spark = get_spark("Simple-ML")
    df = read_students(spark).cache()
    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    train, test = train.cache(), test.cache()
    print(f"Rows: {df.count()}  | train: {train.count()}  | test: {test.count()}  | seed: {SEED}")

    results = {}
    for name, cols in SCENARIOS.items():
        run_linear(train, test, name, cols, results)
    for name, cols in SCENARIOS.items():
        run_logistic(train, test, name, cols, results)

    banner("Summary (measured on the unseen test rows)")
    for name in SCENARIOS:
        lin, log = results[f"linear_{name}"], results[f"logistic_{name}"]
        print(f"{name:<7} linear: R2={lin['r2']:.3f} RMSE={lin['rmse']:.2f} (baseline {lin['baseline_rmse']:.2f})"
              f" | logistic: accuracy={log['accuracy']:.3f} recall(FAIL)={log['recall']:.3f}")
    print("EARLY uses only background information; GRADES also uses G1 and G2.")
    print("An R2 near 0 or below, or an RMSE not below the baseline, means that model is no better")
    print("than guessing the class average. The test set is small, so results vary with the seed.")

    out = OUTPUT_DIR / "csv" / "model_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\nMetrics saved to:", out)
    spark.stop()


if __name__ == "__main__":
    main()
