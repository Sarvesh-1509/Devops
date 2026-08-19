from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="ML prediction API for breast cancer classification",
    version="1.0"
)


# Load trained model
model = joblib.load("models/best_model.pkl")

# Get the feature names used during training
feature_names = model.named_steps["scaler"].feature_names_in_


class PredictionInput(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {
        "message": "Breast Cancer Prediction API is running"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    input_data = pd.DataFrame(
        [data.features],
        columns=feature_names
    )

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }
