import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Pydantic models for request validation
class TransactionData(BaseModel):
    amount: float
    merchant: str
    category: str
    description: Optional[str] = None

# Function to call your Hugging Face Space
def predict_with_hf_space(input_data: Dict[str, Any]) -> Dict[str, Any]:
    # Try these different endpoint structures - you may need to experiment to find the right one
    API_URL = "https://huggingface.co/spaces/bradkim837/fraud-detect-api-bk/api/predict"
    # Or alternative URL formats:
    # API_URL = "https://bradkim837-fraud-detect-api-bk.hf.space/api/predict"
    
    # Get token from environment variable
    hf_token = os.environ.get('HF_API_TOKEN')
    if not hf_token:
        raise HTTPException(status_code=500, detail="HF_API_TOKEN environment variable not set")
        
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json=input_data)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error calling Hugging Face Space: {str(e)}")

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fraud Detection API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: TransactionData):
    # Convert Pydantic model to dict
    input_data = data.dict()
    
    # Get prediction from Hugging Face Space
    try:
        result = predict_with_hf_space(input_data)
        
        # Return the prediction
        return {
            "prediction": result,
            "transaction": input_data
        }
    except HTTPException as e:
        # Re-raise the exception
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
