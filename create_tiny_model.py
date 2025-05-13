# create_tiny_model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Create a tiny RandomForestClassifier
print("Creating tiny model...")
model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)

# Create synthetic data for fitting
X = np.random.rand(10, 5)
y = np.random.randint(0, 2, 10)

# Fit the model
model.fit(X, y)

# Save the model
print("Saving tiny model...")
joblib.dump(model, "model/model_tiny.pkl")

# Create sample feature columns
feature_columns = [f"feature_{i}" for i in range(5)]
joblib.dump(feature_columns, "model/feature_columns.pkl")

# Create sample encoder and scaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, "model/scaler.pkl")

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array([["A"], ["B"], ["C"]]))
joblib.dump(encoder, "model/encoder.pkl")

# Check sizes
print(f"Model size: {os.path.getsize('model/model_tiny.pkl') / 1024:.2f} KB")
print(f"Feature columns size: {os.path.getsize('model/feature_columns.pkl') / 1024:.2f} KB")
print(f"Scaler size: {os.path.getsize('model/scaler.pkl') / 1024:.2f} KB")
print(f"Encoder size: {os.path.getsize('model/encoder.pkl') / 1024:.2f} KB")

print("Done! Created tiny placeholder models for GitHub deployment.")
