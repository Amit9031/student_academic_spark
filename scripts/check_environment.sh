#!/usr/bin/env bash
# Read-only environment check. Installs nothing.
PROJECT="$HOME/StudentPerformanceSpark"
echo "=== macOS ===";   sw_vers | head -n 2; uname -m
echo "=== Java ===";    java -version 2>&1 | head -n 1; echo "JAVA_HOME=$JAVA_HOME"
echo "=== Spark ===";   spark-submit --version 2>&1 | grep -E "version|Scala" | head -n 3
echo "=== Python (venv) ==="
if [ -x "$PROJECT/.venv/bin/python" ]; then
  "$PROJECT/.venv/bin/python" --version
  "$PROJECT/.venv/bin/python" -c "import pyspark,pandas;print('pyspark',pyspark.__version__,'| pandas',pandas.__version__)"
else echo "venv missing: run scripts/setup_environment.sh"; fi
echo "=== Kafka ===";   brew list --versions kafka 2>/dev/null || echo "kafka: not installed"
echo "=== Data ===";    ls -lh "$PROJECT/data/student-mat.csv" "$PROJECT/data/student-por.csv"
