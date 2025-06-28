import os
import pandas as pd

# Always resolve relative to the repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "iris_entity.csv")

def test_data_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] == 300   # Testing if this file has 300 rows

def test_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0
