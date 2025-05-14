#!/bin/bash
# Force reinstall numpy to fix compatibility issues
pip uninstall -y numpy
pip install numpy==1.24.3
