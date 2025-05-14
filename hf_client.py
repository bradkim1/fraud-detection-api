# hf_client.py
import os
import requests

def predict_with_hf_model(input_data):
    API_URL = "https://api-inference.huggingface.co/models/fraud-detect-api-bk"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}"}
    response = requests.post(API_URL, headers=headers, json=input_data)
    return response.json()
