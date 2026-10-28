#!/usr/bin/env bash
# Run the whole project, step by step.   Usage: bash scripts/run_project.sh [all|rdd|sql|ml|dashboard]
cd "$HOME/StudentPerformanceSpark" || exit 1
export JAVA_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
source .venv/bin/activate
QUIET='WARN|incubator|log4j|NativeCodeLoader|require_minimum|Setting default|To adjust'
run() { python "$@" 2>&1 | grep -v -E "$QUIET"; }

step="${1:-all}"
if [[ "$step" == "all" || "$step" == "rdd" ]]; then
  echo "##### STEP 1: RDD operations #####";           run src/rdd/rdd_operations.py
  echo "##### STEP 1b: RDD to DataFrame #####";        run src/rdd/rdd_to_dataframe.py
fi
if [[ "$step" == "all" || "$step" == "sql" ]]; then
  echo "##### STEP 2: DataFrames and Spark SQL #####"; run src/sql/spark_sql_analysis.py --dataframe
  run src/sql/spark_sql_analysis.py --all
  run src/sql/spark_sql_analysis.py --joins
fi
if [[ "$step" == "all" || "$step" == "ml" ]]; then
  echo "##### STEP 3: Machine learning #####";         run src/ml/simple_model.py
fi
if [[ "$step" == "all" || "$step" == "dashboard" ]]; then
  echo "##### STEP 4: Dashboard #####";                run src/dashboard/make_dashboard.py
  open output/dashboard.html 2>/dev/null
fi
