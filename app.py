from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import pandas as pd
import joblib

# Load saved model
model = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")

# Create FastAPI app
app = FastAPI()

# Input schema
class PatientData(BaseModel):
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: str

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value):
        if value not in ["M", "F"]:
            raise ValueError("Gender must be M or F")
        return value

# Home endpoint
@app.get("/")
def home():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):

    input_data = {
        "AGE": data.age,
        "Urea": data.urea,
        "Cr": data.cr,
        "HbA1c": data.hba1c,
        "Chol": data.chol,
        "TG": data.tg,
        "HDL": data.hdl,
        "LDL": data.ldl,
        "VLDL": data.vldl,
        "BMI": data.bmi,
        "Gender_M": 1 if data.gender == "M" else 0
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Match training columns
    df = df.reindex(columns=training_columns, fill_value=0)

    # Predict
    prediction = model.predict(df)

    # Return prediction
    return {
        "prediction": str(prediction[0])
    }