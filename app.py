# In your app.py on Render.com
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class TransactionData(BaseModel):
    amount: float
    merchant: str
    category: str
    description: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fraud Detection API is running"}

@app.post("/predict")
async def predict(data: TransactionData):
    # Call your Hugging Face Space API directly
    HF_SPACE_URL = "https://bradkim837-fraud-detect-api-bk.hf.space/api/predict"  # Adjust endpoint if needed
    
    try:
        # Format the data as expected by your HF Space
        payload = data.dict()
        
        # Make the request to your HF Space
        response = requests.post(HF_SPACE_URL, json=payload)
        response.raise_for_status()
        
        # Return the prediction from your HF Space
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error calling Hugging Face Space: {str(e)}")
