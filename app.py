import os
import sys
import glob
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

def find_file(filename, search_paths=None):
    """Search for a file in multiple directories and return its path if found."""
    if search_paths is None:
        # Default search paths
        search_paths = [
            os.path.dirname(os.path.abspath(__file__)),  # Current script directory
            os.getcwd(),  # Current working directory
            "/opt/render/project/src/",  # Render.com project directory
            "/opt/render/project/",  # Render.com parent directory
            "/tmp/"  # Temporary directory
        ]
    
    # Add parent directories to search paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    search_paths.extend([parent_dir, os.path.dirname(parent_dir)])
    
    # Search each path
    for path in search_paths:
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            return file_path
    
    # Use glob to find the file anywhere in the project directory
    for path in search_paths:
        matches = glob.glob(os.path.join(path, "**", filename), recursive=True)
        if matches:
            return matches[0]
    
    return None

# Load model components
@app.on_event("startup")
async def load_model():
    global model, feature_columns
    try:
        print("Starting model loading...")
        
        # Extensive filesystem debugging
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        print("Files in current directory:")
        for file in os.listdir(os.getcwd()):
            file_path = os.path.join(os.getcwd(), file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                print(f"  - {file} ({file_size:.2f} MB)")
            else:
                print(f"  - {file} (directory)")
        
        # Search for the model files
        model_path = find_file("model.pkl")
        features_path = find_file("feature_columns.pkl")
        
        print(f"Found model path: {model_path}")
        print(f"Found features path: {features_path}")
        
        # Check files exist
        if model_path is None:
            print("Error: Model file not found anywhere in search paths")
            
            # Search for any .pkl files to see what's available
            print("Searching for any .pkl files:")
            for path in [os.getcwd(), os.path.dirname(os.path.abspath(__file__)), "/opt/render/project/src/"]:
                pkls = glob.glob(os.path.join(path, "**", "*.pkl"), recursive=True)
                for pkl in pkls:
                    print(f"  - Found: {pkl}")
            
            return
            
        if features_path is None:
            print("Error: Feature columns file not found anywhere in search paths")
            return
            
        # Check permissions
        print(f"Model file permissions: {oct(os.stat(model_path).st_mode)}")
        print(f"Features file permissions: {oct(os.stat(features_path).st_mode)}")
        
        # Load model and feature columns
        print("Loading model file...")
        try:
            model = joblib.load(model_path)
            print(f"Model loaded successfully: {type(model)}")
        except Exception as model_error:
            print(f"Error loading model: {str(model_error)}")
            import traceback
            traceback.print_exc()
            return
        
        print("Loading feature columns...")
        try:
            feature_columns = joblib.load(features_path)
            print(f"Feature columns loaded successfully: {len(feature_columns)} features")
        except Exception as feature_error:
            print(f"Error loading feature columns: {str(feature_error)}")
            import traceback
            traceback.print_exc()
            return
        
        print("Both model and feature columns loaded successfully!")
        
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
    global model, feature_columns
    return {
        "status": "ok", 
        "message": "Fraud Detection API is running",
        "model_loaded": model is not None,
        "features_loaded": feature_columns is not None,
        "model_type": str(type(model)) if model else None,
        "feature_count": len(feature_columns) if feature_columns else 0
    }

# Debug endpoint
@app.get("/debug")
def debug():
    """Endpoint to debug file system and model loading issues"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    
    # Get list of files in various directories
    base_files = os.listdir(base_dir) if os.path.exists(base_dir) else []
    cwd_files = os.listdir(cwd) if os.path.exists(cwd) else []
    render_files = os.listdir("/opt/render/project/src/") if os.path.exists("/opt/render/project/src/") else []
    
    # Check for model files
    model_path = find_file("model.pkl")
    features_path = find_file("feature_columns.pkl")
    
    return {
        "base_dir": base_dir,
        "cwd": cwd,
        "model_path": model_path,
        "features_path": features_path,
        "base_files": base_files,
        "cwd_files": cwd_files,
        "render_files": render_files,
        "model_exists": os.path.exists(model_path) if model_path else False,
        "features_exists": os.path.exists(features_path) if features_path else False,
        "python_version": sys.version,
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
