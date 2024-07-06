from pandas import read_csv, Series, DataFrame, concat
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("exp_cbr: Importing necessary libraries...")

def model_gen(data_file, model_name):
    print("exp_cbr: Starting model generation...")
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', data_file), 'rb') as file:
        data = pickle.load(file)

    print("exp_cbr: Data loaded, creating CountVectorizer...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(data['combined']).toarray()

    print("exp_cbr: Calculating cosine similarity...")
    similarity = cosine_similarity(vector)

    # Zapisanie do pliku pickle
    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', model_name)
    with open(pickle_file, 'wb') as f:
        pickle.dump(similarity, f)

    print("exp_cbr: Model saved.")

def recommend(game_id, sim_model, data):
    print(f"exp_cbr: Recommending for game_id {game_id}...")
    try:
        index = data[data['name'] == game_id].index[0]

        # Sort the games based on the similarity scores
        sim_scores = sorted(list(enumerate(sim_model[index])), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[0:11]

        game_indices = [i[0] for i in sim_scores]

        print("exp_cbr: Recommendations generated.")
        return data['name'].iloc[game_indices].tolist()
    except Exception as e:
        print(f"exp_cbr: Error in recommend function: {e}")
        return []

print("exp_cbr: Functions defined.")
