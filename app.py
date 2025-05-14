# app.py (or main.py)
from flask import Flask, request, jsonify
from hf_client import predict_with_hf_model

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Fraud Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Preprocess your data if needed
    result = predict_with_hf_model(data)
    # Postprocess the result if needed
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
