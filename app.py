import os
import sys
import joblib
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Global variables for model components
model = None
feature_columns = None
model_found = False

# Load model components
@app.on_event("startup")
async def load_model():
    global model, feature_columns, model_found
    try:
        print("Starting model loading...")
        
        # List of possible model paths to try
        model_paths = [
            "/opt/render/project/src/model/model_minimal.pkl",
            "/opt/render/project/src/model.pkl",
            "/opt/render/project/src/model/model.pkl",
            "model_minimal.pkl",
            "model.pkl"
        ]
        
        # List of possible feature column paths
        feature_paths = [
            "/opt/render/project/src/feature_columns.pkl",
            "/opt/render/project/src/model/feature_columns.pkl",
            "feature_columns.pkl"
        ]
        
        # Try each model path
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Found model at: {model_path}")
                try:
                    model = joblib.load(model_path)
                    print(f"Successfully loaded model: {type(model)}")
                    model_loaded = True
                    model_found = True
                    break
                except Exception as e:
                    print(f"Error loading model from {model_path}: {str(e)}")
        
        if not model_loaded:
            print("Could not load model from any path. Using fallback.")
            # We'll continue without a model
        
        # Try each feature columns path
        features_loaded = False
        for features_path in feature_paths:
            if os.path.exists(features_path):
                print(f"Found feature columns at: {features_path}")
                try:
                    feature_columns = joblib.load(features_path)
                    print(f"Successfully loaded feature columns: {len(feature_columns)} features")
                    features_loaded = True
                    break
                except Exception as e:
                    print(f"Error loading feature columns from {features_path}: {str(e)}")
        
        if not features_loaded:
            print("Could not load feature columns from any path. Using fallback feature list.")
            # Create a minimal set of feature columns for fallback
            feature_columns = ["TransactionAmt", "ProductCD", "card4", "card6", "DeviceType", "DeviceInfo", "M1"]
        
        print(f"Startup complete. Model loaded: {model_loaded}, Features loaded: {features_loaded}")
        
    except Exception as e:
        print(f"Error during model loading: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()

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
    global model, feature_columns, model_found
    return {
        "status": "ok", 
        "message": "Fraud Detection API is running",
        "model_loaded": model is not None,
        "features_loaded": feature_columns is not None,
        "model_found": model_found,
        "model_type": str(type(model)) if model else None,
        "feature_count": len(feature_columns) if feature_columns else 0
    }

# File system debug endpoint
@app.get("/debug")
def debug_info():
    # Get information about the file system
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        
        # Look for model files in various locations
        model_files = []
        for root, dirs, files in os.walk("/opt/render/project"):
            for file in files:
                if file.endswith(".pkl"):
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path) / (1024 * 1024)  # MB
                    model_files.append({"path": full_path, "size_mb": f"{size:.2f}"})
        
        return {
            "base_dir": base_dir,
            "current_dir": cwd,
            "python_version": sys.version,
            "model_loaded": model is not None,
            "model_found": model_found,
            "features_loaded": feature_columns is not None,
            "pkl_files_found": model_files,
            "env_vars": {k: v for k, v in os.environ.items() if not k.startswith("AWS") and not "KEY" in k and not "SECRET" in k}
        }
    except Exception as e:
        return {"error": str(e)}

# Fallback prediction function - returns a prediction even without a model
def predict_fallback(transaction_data):
    # This is a very simple heuristic to provide a fallback:
    # High transaction amounts are more likely to be fraud
    amount = transaction_data.get("TransactionAmt", 0)
    product = transaction_data.get("ProductCD", "")
    
    # Simple rule-based fallback
    fraud_probability = 0.1  # Default low probability
    
    if amount > 1000:
        fraud_probability += 0.2
    
    if product == "W":  # Assuming "W" might be higher risk
        fraud_probability += 0.1
    
    if transaction_data.get("DeviceInfo", "") == "Windows":
        fraud_probability += 0.05
    
    # Cap probability at 0.9
    fraud_probability = min(fraud_probability, 0.9)
    
    # Return prediction (1 if probability > 0.5)
    prediction = 1 if fraud_probability > 0.5 else 0
    
    return {
        "status": "success",
        "prediction": prediction,
        "fraudProbability": fraud_probability,
        "model_version": "fallback v1.0",
        "is_fallback": True
    }

# Prediction endpoint
@app.post("/predict")
async def predict(request: TransactionRequest):
    global model, feature_columns
    
    # Get transaction data
    transaction_data = request.data.dict()
    
    # If model or features aren't loaded, use fallback
    if model is None or feature_columns is None:
        print("Model or features not loaded, using fallback prediction")
        return predict_fallback(transaction_data)
    
    try:
        # Create a DataFrame with zeros for all expected features
        if isinstance(feature_columns, list):
            # If feature_columns is just a list of column names
            prediction_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        else:
            # If feature_columns is something else (like ndarray or Index)
            prediction_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Fill in the values we have from the input
        df = pd.DataFrame([transaction_data])
        for col in df.columns:
            if col in prediction_df.columns:
                prediction_df[col] = df[col].values
        
        # Make prediction
        try:
            prediction = model.predict(prediction_df)[0]
            probability = model.predict_proba(prediction_df)[0][1]
        except Exception as pred_error:
            print(f"Error in model prediction: {str(pred_error)}")
            return predict_fallback(transaction_data)
        
        return {
            "status": "success",
            "prediction": int(prediction),
            "fraudProbability": float(probability),
            "model_version": "v1.0 (minimal)",
            "is_fallback": False
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        
        # Use fallback on error
        return predict_fallback(transaction_data)

# Test endpoint that doesn't require the model
@app.get("/test")
def test():
    return {"message": "API is working correctly"}

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
