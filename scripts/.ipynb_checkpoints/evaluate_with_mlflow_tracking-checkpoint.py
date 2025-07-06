import argparse
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import mlflow

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers.mlflow_dvc_utils import get_git_commit_hash, get_dvc_md5_hash_from_lock  

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to trained model")
parser.add_argument("--test_csv", required=True, help="Path to test CSV")
parser.add_argument("--metrics_out", required=True, help="Path to output metrics JSON")
parser.add_argument("--experiment_name", required=False, help="MLflow experiment name", default=None)
args = parser.parse_args()

# Set MLflow tracking details
mlflow.set_tracking_uri("http://127.0.0.1:8100")  # or your Vertex AI endpoint

# (optional) Try to use SQLITE db and GCS for tracking and artifact storage respectively
# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# # mlflow.set_artifact_uri(os.getenv("MLFLOW_ARTIFACT_URI"))    # This line does not work
# These secrets are available in the ~/.bashrc file. If any error: 1. Source the ~/.bashrc file. or 2. Hardcode the URIs here.

# Set mlflow experiment name 
mlflow.set_experiment(args.experiment_name)

# Load test data
df_test = pd.read_csv(args.test_csv)

# Drop any metadata columns that weren't used during training
df_test = df_test.drop(columns=["flower_id", "event_timestamp"], errors="ignore")

# Define the expected feature columns
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Ensure consistent feature order for inference
X_test = df_test[feature_cols]

# Extract target
y_test = df_test["species"]


# Load model and label encoder
model = joblib.load(args.model)
label_encoder_path = args.model.replace("model", "label_encoder")
label_encoder = joblib.load(label_encoder_path)

# Predict and evaluate
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Save metrics to JSON
os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
with open(args.metrics_out, "w") as f:
    json.dump({
        "accuracy": accuracy,
        "classification_report": report
    }, f, indent=2)

# Log to MLflow (optional)
if args.experiment_name:
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("stage", "evaluation")
        
        # Add details of the exact dvc file and git commit used
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.log_param("test_data_path", args.test_csv)
        mlflow.log_param("test_data_md5", get_dvc_md5_hash_from_lock(args.test_csv))
        
        for label, scores in report.items():
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
