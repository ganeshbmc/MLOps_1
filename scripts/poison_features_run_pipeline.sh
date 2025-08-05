#!/bin/bash

set -e  # Exit on any error
set -x  # Print each command

# -----------------------
# 🧪 Parse CLI arguments
# -----------------------
NOISE_RATIO=0.05
NOISE_STD=5

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --noise_ratio) NOISE_RATIO="$2"; shift ;;
        --noise_std) NOISE_STD="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# -----------------------
# 💉 Poison feature files
# -----------------------
python scripts/poison_features.py \
  --input data/iris_sepal.parquet \
  --output data/iris_sepal.parquet \
  --columns sepal_length sepal_width \
  --noise_ratio "$NOISE_RATIO" \
  --noise_std "$NOISE_STD"

python scripts/poison_features.py \
  --input data/iris_petal.parquet \
  --output data/iris_petal.parquet \
  --columns petal_length petal_width \
  --noise_ratio "$NOISE_RATIO" \
  --noise_std "$NOISE_STD"

# -----------------------
# 📦 Track poisoned files with DVC
# -----------------------
dvc add data/iris_petal.parquet data/iris_sepal.parquet
git add data/iris_petal.parquet.dvc data/iris_sepal.parquet.dvc

# -----------------------
# 📝 Git commit with % and std
# -----------------------
PERCENT=$(awk "BEGIN {printf \"%.1f\", ${NOISE_RATIO} * 100}")
git commit -m "Poisoned ${PERCENT}% of iris features with Gaussian noise (std=${NOISE_STD})"

# -----------------------
# ☁️ Push to DVC remote
# -----------------------
dvc push

# -----------------------
# 🧠 Update Feast registry & materialize
# -----------------------
if [ -d feast_iris/.ipynb_checkpoints ]; then
  echo "Cleaning up Jupyter checkpoint artifacts..."
  rm -r feast_iris/.ipynb_checkpoints
fi

cd feast_iris
feast apply
feast materialize 2025-01-01T00:00:00 2025-12-31T23:59:59
cd ..

# -----------------------
# 🔁 Run DVC pipeline
# -----------------------
dvc repro --force

# -----------------------
# ✅ Final Git commit for pipeline execution
# -----------------------
git add .
git commit -m "Ran pipeline on ${PERCENT}% poisoned features (std=${NOISE_STD}) and logged outputs"
git log --oneline --decorate -n 10
