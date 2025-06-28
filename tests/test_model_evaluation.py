import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_model_accuracy():
    # Load model, label encoder
    MODEL_PATH = os.path.join(ROOT_DIR, "artifacts", "feast_iris_model.joblib")
    LABEL_ENC_PATH = os.path.join(ROOT_DIR, "artifacts", "feast_iris_label_encoder.joblib")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENC_PATH)

    entity_df = pd.read_csv("data/iris_entity.csv")
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], errors='coerce')
    entity_df_2025 = entity_df[entity_df["event_timestamp"].dt.year == 2025]
    true_labels = entity_df_2025[["flower_id", "species"]].drop_duplicates()

    # Load rows for predicting species using model
    ONLINE_FEATURES_PATH = os.path.join(ROOT_DIR, "data", "online_featurs_iris.csv")
    online_features = pd.read_csv(ONLINE_FEATURES_PATH)
    
    # Predict
    X_online = online_features[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    preds = model.predict(X_online)
    decoded_preds = label_encoder.inverse_transform(preds)

    # Attach predictions
    online_features["predicted_label"] = decoded_preds

    # Merge into single table
    merged = online_features.merge(true_labels, on="flower_id", how="inner")

    # Compute accuracy
    accuracy = accuracy_score(merged["species"], merged["predicted_label"])
    
    # Assert that accuracy is above 90%
    assert accuracy > 0.90, f"Model accuracy too low: {acc}"

