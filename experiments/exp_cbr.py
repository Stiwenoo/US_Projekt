import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def model_gen(data_file, model_name):
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', data_file), 'rb') as file:
        data = pickle.load(file)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(data['combined']).toarray()

    similarity = cosine_similarity(vector)

    # Zapisanie do pliku pickle
    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', model_name)
    with open(pickle_file, 'wb') as f:
        pickle.dump(similarity, f)

def recommend(game_title, sim_model, data):
    try:
        if game_title not in data['name'].values:
            return []
        
        index = data[data['name'] == game_title].index[0]
        sim_scores = sorted(list(enumerate(sim_model[index])), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Skip the first one as it is the game itself
        
        game_indices = [i[0] for i in sim_scores]
        return data['name'].iloc[game_indices].tolist()
    except Exception as e:
        print(f"Recommendation error: {e}")
        return []
