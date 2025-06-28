import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import textwrap

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_model_accuracy():
    # Load model, label encoder
    model = joblib.load("artifacts/feast_iris_model.joblib")
    label_encoder = joblib.load("artifacts/feast_iris_label_encoder.joblib")

    entity_df = pd.read_csv("data/iris_entity.csv")
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], errors='coerce')
    entity_df_2025 = entity_df[entity_df["event_timestamp"].dt.year == 2025]
    true_labels = entity_df_2025[["flower_id", "species"]].drop_duplicates()

    # Load rows for predicting species using model
    online_features = pd.read_csv("data/online_features_iris.csv")
    
    # Predict
    X_online = online_features[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    preds = model.predict(X_online)
    decoded_preds = label_encoder.inverse_transform(preds)

    # Attach predictions
    online_features["predicted_label"] = decoded_preds

    # Merge into single table
    merged = online_features.merge(true_labels, on="flower_id", how="inner")
    
    # Print full report
    classifn_report = classification_report(merged["species"], merged["predicted_label"])
    print("Classification Report:\n")
    print(textwrap.indent(classifn_report, prefix="    "))

    conf_matrix = confusion_matrix(merged["species"], merged["predicted_label"])
    print("Confusion Matrix:\n")
    print(textwrap.indent(str(conf_matrix, prefix="    ")))

    # Compute accuracy
    accuracy = accuracy_score(merged["species"], merged["predicted_label"])
    
    # Assert that accuracy is above 90%
    assert accuracy > 0.90, f"Model accuracy too low: {acc}"

