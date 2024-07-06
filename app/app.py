from flask import Flask, request, jsonify
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the model and data
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cbr_model.pkl')
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'rec_games_more.pkl')

with open(MODEL_FILE_PATH, 'rb') as file:
    sim_model = pickle.load(file)

with open(DATA_FILE_PATH, 'rb') as file:
    data = pickle.load(file)


def recommend(game_title, sim_model, data):
    if game_title not in data['name'].values:
        return []

    index = data[data['name'] == game_title].index[0]
    sim_scores = sorted(list(enumerate(sim_model[index])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Skip the first one as it is the game itself

    game_indices = [i[0] for i in sim_scores]
    return data['name'].iloc[game_indices].tolist()


@app.route('/recommend', methods=['GET'])
def get_recommendations():
    game_title = request.args.get('title', default='', type=str)
    if not game_title:
        return jsonify({'error': 'No game title provided'}), 400

    recommendations = recommend(game_title, sim_model, data)
    if not recommendations:
        return jsonify({'error': 'Game title not found'}), 404

    return jsonify({'recommendations': recommendations})


if __name__ == '__main__':
    app.run(debug=True)
