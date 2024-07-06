from flask import Flask, request, jsonify
import os
import pickle
from experiments.exp_cbr import recommend

app = Flask(__name__)

# Load the model and data
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cbr_model.pkl')
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'rec_games_more.pkl')

try:
    with open(MODEL_FILE_PATH, 'rb') as file:
        sim_model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")

try:
    with open(DATA_FILE_PATH, 'rb') as file:
        data = pickle.load(file)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Failed to load data: {e}")

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    try:
        game_title = request.args.get('title', default='', type=str)
        if not game_title:
            return jsonify({'error': 'No game title provided'}), 400
        
        recommendations = recommend(game_title, sim_model, data)
        if not recommendations:
            return jsonify({'error': 'Game title not found'}), 404
        
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
