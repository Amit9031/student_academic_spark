# Student Performance Analytics with PySpark (INT315)

Dataset: UCI Student Performance (`data/student-mat.csv`, semicolon separated, 395 rows).

## One-time setup
```bash
cd ~/StudentPerformanceSpark
source .venv/bin/activate
bash scripts/check_environment.sh
```

## Run everything
```bash
bash scripts/run_project.sh          # all steps, then opens the dashboard
bash scripts/run_project.sh rdd      # or: sql | ml | dashboard
```

## Run each part separately (for the demo)
| Step | Command |
|------|---------|
| RDD operations | `python src/rdd/rdd_operations.py` |
| RDD to DataFrame | `python src/rdd/rdd_to_dataframe.py` |
| DataFrame API | `python src/sql/spark_sql_analysis.py --dataframe` |
| One SQL query (1-26) | `python src/sql/spark_sql_analysis.py --run 15` |
| Export a query to CSV | `python src/sql/spark_sql_analysis.py --run 15 --export` |
| Joins | `python src/sql/spark_sql_analysis.py --joins` |
| Interactive SQL menu | `python src/sql/spark_sql_analysis.py` |
| ML models | `python src/ml/simple_model.py` |
| Dashboard | `python src/dashboard/make_dashboard.py && open output/dashboard.html` |

Add `2>&1 | grep -v -E "WARN|incubator"` to hide harmless warnings.

## Files
- `src/spark_common.py` - shared SparkSession and paths
- `src/rdd/` - RDD transformations/actions, RDD to DataFrame with explicit schema
- `src/sql/` - 26 SQL queries, joins, CSV export, interactive console
- `src/ml/simple_model.py` - Linear Regression (G3) and Logistic Regression (pass/fail)
- `src/dashboard/make_dashboard.py` - builds `output/dashboard.html`
- `output/csv/` - exported results and `model_metrics.json`

## ML notes
- G3 is never a feature. EARLY scenario uses no G1/G2; GRADES scenario adds them.
- Models are fitted on an 80% training split (seed 42) and scored on the unseen 20%.
- FAIL = G3 < 10 is the positive class for precision/recall.
