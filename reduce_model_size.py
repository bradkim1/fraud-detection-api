# reduce_model_size.py
import joblib
import os
import sys

print("Starting model size reduction...")

# Check if model file exists
if not os.path.exists("model/model.pkl"):
    print("Error: model/model.pkl not found")
    sys.exit(1)

# Get original file size
original_size_mb = os.path.getsize("model/model.pkl") / (1024 * 1024)
print(f"Original model size: {original_size_mb:.2f} MB")

# Load the original model
print("Loading model...")
model = joblib.load("model/model.pkl")

# Check model type
model_type = type(model).__name__
print(f"Model type: {model_type}")

# Reduce model size based on type
if hasattr(model, 'n_estimators'):
    print(f"Current n_estimators: {model.n_estimators}")
    # Save original number of estimators
    original_n_estimators = model.n_estimators
    
    # Reduce to just 10 trees
    model.n_estimators = 10
    print(f"Reduced n_estimators to: {model.n_estimators}")
    
    # Save smaller model
    print("Saving reduced model...")
    joblib.dump(model, "model/model_small.pkl")
    
    # Check new file size
    new_size_mb = os.path.getsize("model/model_small.pkl") / (1024 * 1024)
    print(f"New model size: {new_size_mb:.2f} MB")
    print(f"Size reduction: {(1 - new_size_mb/original_size_mb)*100:.2f}%")
    
    print("\nNote: This reduced model has fewer trees and may be less accurate.")
    print("Update your code to use 'model_small.pkl' instead of 'model.pkl'")
else:
    print("Model doesn't have n_estimators attribute. Cannot reduce size using this method.")
    print("Try a different approach for size reduction.")
