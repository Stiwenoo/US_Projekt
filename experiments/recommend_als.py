import pickle
import os
# import pandas as pd
# import numpy as np
# import implicit


def model_load(model_file):
    with open(os.path.join(os.path.dirname(__file__), '..', 'models', model_file), 'rb') as file:
        model = pickle.load(file)

    return model


def mapping_game_titles(pivot_name):
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', pivot_name), 'rb') as file:
        pivot_table = pickle.load(file)

    # Tworzymy słownik mapujący indeksy z pivot_table do macierzy rzadkiej
    index_to_app_id = {i: app_id for i, app_id in enumerate(pivot_table.index)}

    return index_to_app_id


def recommend_game_from_game(model, game_id, number_of_recommendations, map_of_titles):

    # To do dokonczenia
    if type(game_id) is str:
        game_id = list(map_of_titles.keys())[list(map_of_titles.values()).index(game_id)]

    print('app_id:', game_id, '\nGame:', map_of_titles[game_id])
    # print('\n', map_of_titles)
    ids, scores = model.similar_items(game_id, number_of_recommendations)
    print(scores.tolist())

    mapped_list = [map_of_titles[value] for value in ids]

    return mapped_list


# print(mapping_game_titles('pivot_9k_gamers.pkl'))
print(recommend_game_from_game(model_load('als_model_9k_gamers.pkl'), 'Dota 2', 10,
                               mapping_game_titles('pivot_9k_gamers.pkl')))
