import os
import pandas as pd

def test_data_shape():
    df = pd.read_csv("data/iris_entity.csv")
    assert df.shape[0] == 300   # Testing if this file has 300 rows

def test_no_missing_values():
    df = pd.read_csv("data/iris_entity.csv")
    assert df.isnull().sum().sum() == 0
