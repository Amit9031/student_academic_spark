#!/usr/bin/env bash
# One-time setup. Safe to re-run: it only installs what is missing.
set -e
PROJECT="$HOME/StudentPerformanceSpark"
cd "$PROJECT"
mkdir -p src/{rdd,sql,streaming,ml,visualization} scripts output/{csv,models,figures,streaming,checkpoints} docs

JAVA21="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
[ -d "$JAVA21" ] || { echo "Install Java 21: brew install openjdk@21"; exit 1; }
export JAVA_HOME="$JAVA21"; export PATH="$JAVA_HOME/bin:$PATH"

[ -d .venv ] || /opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
command -v kafka-topics >/dev/null || brew install kafka
echo "Setup complete. Run: bash scripts/check_environment.sh"
