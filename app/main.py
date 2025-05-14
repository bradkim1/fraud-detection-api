# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# At your root endpoint, add support for HEAD method
@app.get("/")
@app.head("/")  # Add this line to support HEAD requests
def read_root():
    return {"status": "ok", "message": "Fraud Detection API is running"}
# Add this endpoint
@app.get("/test")
def test_endpoint():
    return {
        "status": "ok",
        "message": "Test endpoint is working",
        "timestamp": str(datetime.datetime.now())
    }
import platform
import datetime

@app.get("/health")
def health_check():
    # Check if the model directory exists
    model_dir_exists = os.path.exists("/opt/render/project/src/model")
    
    # Check if feature_columns.pkl exists
    feature_columns_exists = os.path.exists("/opt/render/project/src/model/feature_columns.pkl")
    
    # List files in the model directory
    model_files = os.listdir("/opt/render/project/src/model") if model_dir_exists else []
    
    return {
        "status": "healthy",
        "model_approach": "rules-based",
        "version": "1.0",
        "environment": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "processor": platform.processor()
        },
        "diagnostics": {
            "model_dir_exists": model_dir_exists,
            "feature_columns_exists": feature_columns_exists,
            "model_files": model_files,
            "current_time": str(datetime.datetime.now())
        }
    }
@app.post("/predict")
async def predict(transaction: TransactionData):
    try:
        # Get transaction data
        data = transaction.data
        
        # Handle potential missing or invalid fields
        try:
            amt = float(data.get("TransactionAmt", 0))
        except (ValueError, TypeError):
            amt = 0
            
        product = str(data.get("ProductCD", ""))
        card_type = str(data.get("card4", ""))
        card_category = str(data.get("card6", ""))
        device_type = str(data.get("DeviceType", ""))
        device_info = str(data.get("DeviceInfo", ""))
        m1 = str(data.get("M1", "T"))
        
        # Rule-based factors
        base_prob = 0.05
        
        # Amount factor (higher amounts = higher risk)
        if amt > 1000:
            amt_factor = 0.3
        elif amt > 500:
            amt_factor = 0.15
        else:
            amt_factor = 0.05
            
        # Product code factor
        product_factors = {"W": 0.2, "C": 0.05, "H": 0.1, "S": 0.15, "R": 0.03}
        product_factor = product_factors.get(product, 0.05)
        
        # Card type factor
        if card_type.lower() == "american express":
            card_factor = 0.1
        elif card_type.lower() == "visa" and card_category.lower() == "credit":
            card_factor = 0.05
        else:
            card_factor = 0.02
            
        # Device factor
        if device_type.lower() == "mobile" and device_info.lower() == "android":
            device_factor = 0.15
        elif device_type.lower() == "desktop" and device_info.lower() != "windows":
            device_factor = 0.05
        else:
            device_factor = 0.01
            
        # M1 flag factor
        m1_factors = {"T": 0.02, "F": 0.15, "M": 0.08}
        m1_factor = m1_factors.get(m1, 0.02)
        
        # Add a small random variation
        import random
        random_factor = random.uniform(-0.05, 0.05)
        
        # Calculate final probability
        fraud_prob = base_prob + amt_factor + product_factor + card_factor + device_factor + m1_factor + random_factor
        
        # Ensure probability is between 0 and 1
        fraud_prob = max(0, min(0.95, fraud_prob))
        
        # Create risk factors explanation
        risk_factors = []
        if amt > 1000:
            risk_factors.append("High transaction amount")
        if product == "W":
            risk_factors.append("High-risk product category")
        if card_type.lower() == "american express":
            risk_factors.append("Card type associated with higher fraud rates")
        if device_type.lower() == "mobile" and device_info.lower() == "android":
            risk_factors.append("Device type associated with higher fraud rates")
        if m1 == "F":
            risk_factors.append("Suspicious M1 flag value")
        
        return {
            "prediction": 1 if fraud_prob > 0.5 else 0,
            "fraudProbability": float(fraud_prob),
            "model_version": "v1.0-rules",
            "risk_factors": risk_factors,
            "timestamp": str(datetime.datetime.now())
        }
    
    except Exception as e:
        # Log the error
        print(f"Prediction error: {str(e)}")
        
        # Return a graceful error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction failed",
                "message": str(e),
                "timestamp": str(datetime.datetime.now())
            }
        )
from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Print debugging information
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir("."))
print("Files in parent directory:", os.listdir("..") if os.path.exists("..") else "Cannot access parent directory")
print("Files in /opt/render/project/src:", os.listdir("/opt/render/project/src") if os.path.exists("/opt/render/project/src") else "Cannot access directory")
if os.path.exists("/opt/render/project/src/model"):
    print("Files in /opt/render/project/src/model:", os.listdir("/opt/render/project/src/model"))
else:
    print("Cannot access /opt/render/project/src/model - directory doesn't exist")
# Add this function to your main.py
def create_files_if_missing():
    print("Checking for missing model files...")
    
    # Define model directory
    model_dir = "/opt/render/project/src/model"
    
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        print(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
    
    # Create feature_columns.pkl if it doesn't exist
    feature_columns_path = os.path.join(model_dir, "feature_columns.pkl")
    if not os.path.exists(feature_columns_path):
        print(f"Creating {feature_columns_path}")
        import joblib
        feature_columns = [
            "TransactionAmt", "ProductCD_W", "ProductCD_C", "ProductCD_H", "ProductCD_S", "ProductCD_R",
            "card4_visa", "card4_mastercard", "card4_american express", "card4_discover",
            "card6_credit", "card6_debit", "card6_charge",
            "DeviceType_desktop", "DeviceType_mobile",
            "M1_T", "M1_F", "M1_M"
        ]
        joblib.dump(feature_columns, feature_columns_path)
        print(f"Created {feature_columns_path}")
    
    # Create other files that might be needed...
    print("File check complete.")

# Call this function at the top of your app initialization
create_files_if_missing()

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
# Define model paths with flexible base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
# Alternative model directory for Render
RENDER_MODEL_DIR = "/opt/render/project/src/model"

# Check which directory exists and use that
if os.path.exists(RENDER_MODEL_DIR):
    MODEL_DIR = RENDER_MODEL_DIR
    print(f"Using Render model directory: {MODEL_DIR}")
else:
    print(f"Using local model directory: {MODEL_DIR}")

# Define file paths
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
    }

# Prediction endpoint

# In your prediction endpoint
@app.post("/predict")
async def predict(transaction: TransactionData):
    # Always use the fallback prediction method
    # Get transaction data
    data = transaction.data
    
    # Calculate a simplified fraud score
    amt = float(data.get("TransactionAmt", 0))
    product = data.get("ProductCD", "")
    card_type = data.get("card4", "")
    device_type = data.get("DeviceType", "")
    
    # Simple rules-based score
    base_prob = 0.05
    amt_factor = 0.3 if amt > 1000 else (0.15 if amt > 500 else 0.05)
    product_factor = 0.2 if product == "W" else 0.05
    card_factor = 0.1 if card_type == "american express" else 0.02
    device_factor = 0.15 if device_type == "mobile" else 0.01
    m1 = data.get("M1", "T")
    m1_factor = 0.15 if m1 == "F" else (0.08 if m1 == "M" else 0.02)
    
    # Calculate probability
    import random
    fraud_prob = base_prob + amt_factor + product_factor + card_factor + device_factor + m1_factor + random.uniform(-0.05, 0.05)
    fraud_prob = max(0, min(0.95, fraud_prob))
    
    return {
        "prediction": 1 if fraud_prob > 0.5 else 0,
        "fraudProbability": fraud_prob,
        "model_version": "v1.0-simplified",
        "note": "Using simplified prediction logic for demo purposes"
    }
# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
