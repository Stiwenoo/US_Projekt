from flask import Flask, request, jsonify, render_template
import os
import pickle
import sys
import threading

# Dodaj katalog nadrzędny do sys.path, aby móc importować moduły z innych katalogów
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importuj bezpośrednio z exp_cbr.py
from experiments.exp_cbr import recommend, model_gen

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
    except FileNotFoundError:
        print("app: Model file not found. Generating model...")
        # Wywołaj funkcję model_gen do wygenerowania modelu
        model_gen(DATA_FILE_PATH, MODEL_FILE_PATH)
        load_model()  # Rekurencyjnie próbuj załadować model ponownie

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

# Funkcja pomocnicza do znalezienia podobnych tytułów gier
def find_similar_titles(input_title, all_titles):
    input_title_lower = input_title.lower()
    similar_titles = difflib.get_close_matches(input_title_lower, all_titles, n=3, cutoff=0.6)
    return similar_titles

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

        # Sprawdzamy, czy istnieje dokładny tytuł gry
        exact_match = data[data['name'].str.lower() == game_title.lower()]
        if not exact_match.empty:
            recommendations = recommend(game_title, sim_model, data)
            if not recommendations:
                return jsonify({'error': 'Game title not found'}), 404
            return render_template('index.html', recommendations=recommendations)
        
        # Jeśli nie znaleziono dokładnego dopasowania, szukamy podobnych tytułów
        all_game_titles = data['name'].str.lower().tolist()
        similar_titles = find_similar_titles(game_title, all_game_titles)
        if similar_titles:
            return jsonify({'error': 'Game title not found. Did you mean one of these?', 'similar_titles': similar_titles})
        else:
            return jsonify({'error': 'Game title not found and no similar titles found'}), 404

    except Exception as e:
        print(f"app: Error in get_recommendations: {e}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
