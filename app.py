import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Global variables for model components
model = None
feature_columns = None

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal Server Error: {str(exc)}"},
    )

# Load model components
@app.on_event("startup")
async def load_model():
    global model, feature_columns
    try:
        print("Starting model loading...")
        # Determine base directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Model paths
        model_path = os.path.join(BASE_DIR, "model.pkl")
        features_path = os.path.join(BASE_DIR, "feature_columns.pkl")
        
        print(f"Model path: {model_path}")
        print(f"Features path: {features_path}")
        
        # Check files exist
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
            
        if not os.path.exists(features_path):
            print(f"Error: Feature columns file not found at {features_path}")
            return
            
        # Load model and feature columns
        print("Loading model file...")
        model = joblib.load(model_path)
        print(f"Model loaded successfully: {type(model)}")
        
        print("Loading feature columns...")
        feature_columns = joblib.load(features_path)
        print(f"Feature columns loaded successfully: {len(feature_columns)} features")
        
    except Exception as e:
        print(f"Error during model loading: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        # Don't raise an exception - let the app continue
        # We'll handle missing model in the endpoints

# Data models
class TransactionData(BaseModel):
    TransactionAmt: float
    ProductCD: str
    card4: str
    card6: str
    DeviceType: str
    DeviceInfo: str
    M1: str

# Request model
class TransactionRequest(BaseModel):
    data: TransactionData

# Health check endpoint
@app.get("/")
def health_check():
    global model, feature_columns
    return {
        "status": "ok", 
        "message": "Fraud Detection API is running",
        "model_loaded": model is not None,
        "features_loaded": feature_columns is not None
    }

# Prediction endpoint
@app.post("/predict")
async def predict(request: TransactionRequest):
    global model, feature_columns
    
    # Check if model is loaded
    if model is None or feature_columns is None:
        return {
            "status": "error",
            "message": "Model not loaded properly. Check server logs.",
            "fallback_prediction": {
                "prediction": 0,
                "fraudProbability": 0.01,
                "model_version": "fallback v1.0"
            }
        }
    
    try:
        # Get transaction data
        transaction_data = request.data.dict()
        
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
            "status": "success",
            "prediction": int(prediction),
            "fraudProbability": float(probability),
            "model_version": "v1.0"
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": f"Error making prediction: {str(e)}",
            "fallback_prediction": {
                "prediction": 0,
                "fraudProbability": 0.5,
                "model_version": "fallback v1.0"
            }
        }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
