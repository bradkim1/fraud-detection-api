# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import os

# Set up the FastAPI app
app = FastAPI(title="Fraud Detection API")

# Enable CORS to allow requests from Hugging Face Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this to your HF Space URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model_minimal.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

# Load model components
try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {str(e)}")
    MODEL_LOADED = False

# Define input data model
class TransactionData(BaseModel):
    data: Dict[str, Any]

# Root endpoint
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fraud Detection API is running"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED
    }

# Prediction endpoint
@app.post("/predict")
async def predict(transaction: TransactionData):
    if not MODEL_LOADED:
        # For bootcamp purposes, use a simplified prediction if model doesn't load
        data = transaction.data
        # Calculate a simplified fraud score
        amt = data.get("TransactionAmt", 0)
        product = data.get("ProductCD", "")
        card_type = data.get("card4", "")
        device_type = data.get("DeviceType", "")
        
        # Simple rules-based score
        base_prob = 0.05
        amt_factor = 0.3 if amt > 1000 else (0.15 if amt > 500 else 0.05)
        product_factor = 0.2 if product == "W" else 0.05
        card_factor = 0.1 if card_type == "american express" else 0.02
        device_factor = 0.15 if device_type == "mobile" else 0.01
        
        # Calculate probability
        import random
        fraud_prob = base_prob + amt_factor + product_factor + card_factor + device_factor + random.uniform(-0.05, 0.05)
        fraud_prob = max(0, min(0.95, fraud_prob))
        
        return {
            "prediction": 1 if fraud_prob > 0.5 else 0,
            "fraudProbability": fraud_prob,
            "model_version": "v1.0-simplified",
            "note": "Using simplified prediction logic as model could not be loaded"
        }
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([transaction.data])
        
        # For a real implementation, you would:
        # 1. Process the data (handle missing values, encode categoricals, etc.)
        # 2. Ensure the DataFrame has all required features
        # 3. Align feature columns with what the model expects
        
        # For bootcamp demo, we'll use a simplified approach:
        prediction_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Fill in values we have
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
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
