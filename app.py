import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Global variables for model components
model = None
feature_columns = None

# Load model components
@app.on_event("startup")
async def load_model():
    global model, feature_columns
    try:
        # Determine base directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Load model and columns
        model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
        feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")

# Pydantic models
class TransactionData(BaseModel):
    TransactionAmt: float
    ProductCD: str
    card4: str
    card6: str
    DeviceType: str
    DeviceInfo: str
    M1: str

class TransactionRequest(BaseModel):
    data: TransactionData

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fraud Detection API is running"}

# Prediction function (adapted from your Streamlit app)
def predict_fraud_direct(transaction_data):
    try:
        # Create a DataFrame with zeros for all expected features
        prediction_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Fill in the values we have from the input
        df = pd.DataFrame([transaction_data])
        for col in df.columns:
            if col in prediction_df.columns:
                prediction_df[col] = df[col].values
        
        # Make prediction
        prediction = model.predict(prediction_df)[0]
        probability = model.predict_proba(prediction_df)[0][1]
        
        return {
            "prediction": int(prediction),
            "fraudProbability": float(probability),
            "model_version": "v1.0"
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Prediction endpoint
@app.post("/predict")
async def predict(request: TransactionRequest):
    return predict_fraud_direct(request.data.dict())

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
