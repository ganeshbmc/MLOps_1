import os
import pandas as pd

def test_data_shape():
    df = pd.read_csv("data/iris_entity.csv")
    print(f"\nLoaded data with shape: {df.shape}")
    assert df.shape[0] == 300   # Testing if this file has 300 rows

def test_no_missing_values():
    df = pd.read_csv("data/iris_entity.csv")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nMissing values in data file: {missing}")
    assert total_missing == 0