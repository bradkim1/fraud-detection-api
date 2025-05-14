#!/bin/bash
echo "Starting custom build process..."

# Upgrade pip
pip install --upgrade pip

# Uninstall and reinstall packages with exact versions
pip uninstall -y numpy scikit-learn pandas joblib
pip install numpy==2.2.5
pip install scikit-learn==1.6.1
pip install pandas==2.2.3
pip install joblib==1.5.0

# Install the rest of the requirements
pip install -r requirements.txt

echo "Build process completed."
