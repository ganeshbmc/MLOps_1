import argparse
import os
import json
import joblib
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, classification_report

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to trained model")
parser.add_argument("--test_csv", required=True, help="Path to test CSV")
parser.add_argument("--metrics_out", required=True, help="Path to output metrics JSON")
parser.add_argument("--experiment_name", required=False, help="MLflow experiment name", default=None)
args = parser.parse_args()

# Load test data
df_test = pd.read_csv(args.test_csv)
X_test = df_test.drop(columns=["species"])
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
        for label, scores in report.items():
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
