from flask import Flask, request, jsonify, render_template
import os
import pickle
import sys
import threading

# Dodaj katalog nadrzędny do sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("app: Importing recommend function...")
try:
    from experiments.exp_cbr import recommend
    print("app: Recommend function imported successfully.")
except Exception as e:
    print(f"app: Error importing recommend function: {e}")

app = Flask(__name__)

MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cbr_model.pkl')
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'rec_games_more.pkl')

# Globalne zmienne do przechowywania modelu i danych
sim_model = None
data = None

def load_model():
    global sim_model
    try:
        print("app: Loading model...")
        with open(MODEL_FILE_PATH, 'rb') as file:
            print("app: Model file opened.")
            sim_model = pickle.load(file)
            print("app: Model loaded.")
    except Exception as e:
        print(f"app: Failed to load model: {e}")

def load_data():
    global data
    try:
        print("app: Loading data...")
        with open(DATA_FILE_PATH, 'rb') as file:
            print("app: Data file opened.")
            data = pickle.load(file)
            print("app: Data loaded.")
    except Exception as e:
        print(f"app: Failed to load data: {e}")

# Uruchom wczytywanie modelu i danych w wątkach
threading.Thread(target=load_model).start()
threading.Thread(target=load_data).start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    try:
        game_title = request.args.get('title', default='', type=str)
        if not game_title:
            return jsonify({'error': 'No game title provided'}), 400

        if sim_model is None or data is None:
            return jsonify({'error': 'Model or data not loaded yet'}), 503

        print(f"app: Received recommendation request for game: {game_title}")
        recommendations = recommend(game_title, sim_model, data)
        if not recommendations:
            return jsonify({'error': 'Game title not found'}), 404

        return render_template('index.html', recommendations=recommendations)
    except Exception as e:
        print(f"app: Error in get_recommendations: {e}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
