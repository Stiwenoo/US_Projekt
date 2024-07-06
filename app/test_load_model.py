import os
import pickle

MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cbr_model.pkl')

try:
    print("Loading model...")
    with open(MODEL_FILE_PATH, 'rb') as file:
        sim_model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
