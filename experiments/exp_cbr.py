from pandas import read_csv, Series, DataFrame, concat
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


# model_gen('rec_games_more.pkl', 'cbr_model.pkl')


def recommend(game_id, model_file):
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', model_file), 'rb') as file:
        sim_model = pickle.load(file)

    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'rec_games_more.pkl'), 'rb') as file:
        data = pickle.load(file)

    index = data[data['name'] == game_id].index[0]

    # Sort the games based on the similarity scores
    sim_scores = sorted(list(enumerate(sim_model[index])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:11]

    game_indices = [i[0] for i in sim_scores]

    print(data['name'].iloc[game_indices].tolist())


# model_gen('rec_games_more.pkl', 'cbr_model.pkl')
recommend('RimWorld', 'cbr_model.pkl')
