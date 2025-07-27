from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Iris Classifier API")

# Load model
model = joblib.load("feast_iris_model.joblib")

# Load label encoder
label_encoder = joblib.load("feast_iris_label_encoder.joblib") 

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
@app.get("/")
def root():
    return {"message": "Welcome to the Iris API. Whatever you post is classified!"}

@app.post("/predict/")
def predict_iris_class(data: IrisInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    return {"The iris species that you bringeth is": predicted_class}