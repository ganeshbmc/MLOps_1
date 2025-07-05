import argparse
import os
import json
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", required=True, help="Path to training CSV")
parser.add_argument("--model_out", required=True, help="Path to save the trained model")
parser.add_argument("--metrics_out", required=True, help="Path to save metrics JSON")
parser.add_argument("--experiment_name", required=True, help="MLflow experiment name")
args = parser.parse_args()

# Load data
df = pd.read_csv(args.train_csv)
X = df.drop(columns=["species"])
y = df["species"]

# Label encode the target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Set MLflow experiment
mlflow.set_experiment(args.experiment_name)

with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 4, 6]
    }
    clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring="accuracy")
    clf.fit(X, y_encoded)

    # Log individual trial results
    for i, params in enumerate(clf.cv_results_['params']):
        mlflow.log_params({f"trial_{i}_{k}": v for k, v in params.items()})
        mlflow.log_metric(f"trial_{i}_mean_test_score", clf.cv_results_['mean_test_score'][i])

    # Log best model info
    mlflow.log_params(clf.best_params_)
    mlflow.log_metric("best_cv_score", clf.best_score_)

    # Save best model and label encoder
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(clf.best_estimator_, args.model_out)
    label_path = args.model_out.replace("model", "label_encoder")
    joblib.dump(label_encoder, label_path)

    # Save metrics JSON
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump({
            "best_cv_score": clf.best_score_,
            "best_params": clf.best_params_
        }, f)

    # Log model in MLflow
    mlflow.sklearn.log_model(clf.best_estimator_, "model")

print(f"Training complete. Model saved to {args.model_out}")
