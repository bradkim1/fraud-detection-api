# create_minimal_model.py
import joblib
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Starting minimal model creation...")

# Check if model file exists
if not os.path.exists("model/model.pkl"):
    print("Error: model/model.pkl not found")
    sys.exit(1)

# Get original file size
original_size_mb = os.path.getsize("model/model.pkl") / (1024 * 1024)
print(f"Original model size: {original_size_mb:.2f} MB")

# Load the original model
print("Loading original model...")
original_model = joblib.load("model/model.pkl")

# Extract key properties from the original model
print("Extracting key properties...")
if isinstance(original_model, RandomForestClassifier):
    # Create a new, minimal RandomForestClassifier
    print("Creating minimal RandomForestClassifier...")
    minimal_model = RandomForestClassifier(
        n_estimators=5,  # Very few trees
        max_depth=5,     # Shallow trees
        min_samples_split=5,
        random_state=42
    )
    
    # Get a sample of training data if available
    if hasattr(original_model, 'n_features_in_'):
        n_features = original_model.n_features_in_
        print(f"Number of features: {n_features}")
        
        # Create synthetic data for fitting
        X_synthetic = np.random.rand(100, n_features)
        y_synthetic = np.random.randint(0, 2, 100)
        
        # Fit the minimal model
        print("Fitting minimal model on synthetic data...")
        minimal_model.fit(X_synthetic, y_synthetic)
        
        # Save the minimal model
        print("Saving minimal model...")
        joblib.dump(minimal_model, "model/model_minimal.pkl")
        
        # Check new file size
        new_size_mb = os.path.getsize("model/model_minimal.pkl") / (1024 * 1024)
        print(f"New model size: {new_size_mb:.2f} MB")
        print(f"Size reduction: {(1 - new_size_mb/original_size_mb)*100:.2f}%")
        
        print("\nIMPORTANT: This is a placeholder model for GitHub deployment.")
        print("It will NOT provide accurate predictions! Use only for demonstration purposes.")
    else:
        print("Could not determine number of features. Unable to create minimal model.")
else:
    print(f"Model type {type(original_model).__name__} is not supported for this reduction method.")

# Create a dummy feature columns file if it doesn't exist
if os.path.exists("model/feature_columns.pkl"):
    feature_columns = joblib.load("model/feature_columns.pkl")
    print(f"Loaded feature_columns with {len(feature_columns)} features")
else:
    print("Creating dummy feature_columns.pkl")
    feature_columns = [f"feature_{i}" for i in range(n_features)]
    joblib.dump(feature_columns, "model/feature_columns.pkl")
