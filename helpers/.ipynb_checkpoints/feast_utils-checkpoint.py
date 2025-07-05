from feast import FeatureStore
import pandas as pd
from datetime import datetime

def load_training_data(from_year: int, to_year: int, entity_path="data/iris_entity.csv") -> pd.DataFrame:
    """
    Load training data using Feast's offline (historical) feature store interface.

    This function reads an entity dataframe (including timestamp and label),
    filters it by a specified year range, and retrieves historical features
    defined in the sepal and petal feature views using Feast.

    Parameters
    ----------
    from_year : int
        The starting year (inclusive) for filtering the entity dataframe.
    to_year : int
        The ending year (inclusive) for filtering the entity dataframe.
    entity_path : str, optional
        Path to the CSV file containing the entity dataframe, which must
        include 'flower_id', 'event_timestamp', and 'species' columns.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the joined entity and feature values including:
        - flower_id
        - event_timestamp
        - species (label)
        - sepal and petal features from Feast
    """
    
    store = FeatureStore(repo_path="feast_iris")
    entity_df = pd.read_csv(entity_path, parse_dates=["event_timestamp"])

    entity_df = entity_df[
        (entity_df["event_timestamp"].dt.year >= from_year) &
        (entity_df["event_timestamp"].dt.year <= to_year)
    ]

    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "sepal_features:sepal_length",
            "sepal_features:sepal_width",
            "petal_features:petal_length",
            "petal_features:petal_width"
        ]
    ).to_df()

    return feature_df  # Keep flower_id, timestamp, label, features




def load_simulated_online_features(from_year: int, to_year: int, entity_path="data/iris_entity.csv") -> pd.DataFrame:
    """
    Simulate online inference by retrieving features using Feast's online store API.

    This function mimics a real-time inference scenario by selecting entity IDs
    from a CSV file based on a time range (e.g., year == 2025), retrieving online
    features from the SQLite-based online store, and joining with ground truth labels.

    Parameters
    ----------
    from_year : int
        The starting year (inclusive) to select entity rows for simulation.
    to_year : int
        The ending year (inclusive) to select entity rows for simulation.
    entity_path : str, optional
        Path to the CSV file containing the entity dataframe, which must
        include 'flower_id', 'event_timestamp', and 'species' columns.

    Returns
    -------
    pd.DataFrame
        A dataframe containing:
        - flower_id
        - sepal and petal online features retrieved via Feast
        - species (label) for evaluation purposes
    """
    
    store = FeatureStore(repo_path="feast_iris")
    
    df = pd.read_csv(entity_path, parse_dates=["event_timestamp"])
    df = df[
        (df["event_timestamp"].dt.year >= from_year) &
        (df["event_timestamp"].dt.year <= to_year)
    ]

    entity_ids = df["flower_id"].tolist()

    online_features = store.get_online_features(
        features=[
            "sepal_features:sepal_length",
            "sepal_features:sepal_width",
            "petal_features:petal_length",
            "petal_features:petal_width"
        ],
        entity_rows=[{"flower_id": i} for i in entity_ids]
    ).to_df()

    # Join features with labels
    result = pd.merge(df, online_features, on="flower_id", how="left")
    return result
