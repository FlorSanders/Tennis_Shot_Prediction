import os
import sys

# Enable importing from src directory
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

# Constants
data_path = os.path.abspath(os.path.join(base_path, "..", "data"))
models_path = os.path.abspath(os.path.join(base_path, "..", "models", "data"))
