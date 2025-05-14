import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import Dict, Any, Optional

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Pydantic models for request validation
class TransactionData(BaseModel):
    # Add fields relevant to your fraud detection model
    # Examples (modify based on your actual model's requirements):
    amount: float
    merchant: str
    category: str
    description: Optional[str] = None
    # Add any other fields your model expects

class PredictionResponse(BaseModel):
    prediction: Dict[str, Any]
    score: float

# Hugging Face client function
def predict_with_hf_model(input_data: Dict[str, Any]) -> Dict[str, Any]:
    # Replace with your actual model ID
    API_URL = "https://api-inference.huggingface.co/models/your-model-id"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json=input_data)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error calling Hugging Face API: {str(e)}")

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fraud Detection API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: TransactionData):
    # Convert Pydantic model to dict
    input_data = data.dict()
    
    # Get prediction from Hugging Face
    hf_result = predict_with_hf_model(input_data)
    
    # Process the HF result as needed
    # This is just an example - adapt to your actual model's output format
    fraud_score = hf_result.get("score", 0.0)
    
    return {
        "prediction": hf_result,
        "score": fraud_score
    }

# Add this for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
