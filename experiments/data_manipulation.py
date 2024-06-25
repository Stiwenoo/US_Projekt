import pandas as pd
import os
import pickle


def dataframe_gen():
    games_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'games.csv'))
    rec_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'recommendations.csv'))

    # Usunięcie kolumn
    columns_to_drop_r = ['helpful', 'funny', 'is_recommended', 'review_id']
    rec_data.drop(columns=columns_to_drop_r, inplace=True)

    # Usunięcie kolumn
    columns_to_drop_g = ['date_release','win','mac','linux','rating','positive_ratio','user_reviews','price_final','price_original','discount','steam_deck']
    games_data.drop(columns=columns_to_drop_g, inplace=True)

    # Mergowanie
    merged_data = pd.merge(rec_data, games_data, on='app_id', how='inner')
    print(len(merged_data))
    print(merged_data.head())

    merged_data_unique = merged_data.drop_duplicates(subset=['app_id', 'user_id'])
    print(len(merged_data_unique))
    print(merged_data_unique.head())

    # Zapisanie do pliku pickle
    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'rec_games.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(merged_data_unique, f)

    return merged_data_unique


def data_sampling(data_file: str, column_tobe_sampled: str, val1: int = 0, val2: int = 9000):
    # Wczytywanie danych
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', data_file), 'rb') as file:
        data = pickle.load(file)

    print('Dlugosc danych: '+ str(len(data)))

    # Wybieranie próbki zawierającej 100 tysięcy unikalnych użytkowników
    # column_sample = merged_data_unique.sample(n=100000, random_state=1)

    # Filtrowanie użytkowników (user_id) - na przykład wybór pierwszych X unikalnych użytkowników
    column_sample = data[column_tobe_sampled].unique()[val1:val2]
    print(column_sample)

    # Filtrowanie danych do wybranych użytkowników
    data_sampled = data[data[column_tobe_sampled].isin(column_sample)]
    print('Dlugosc probki: ' + str(len(data_sampled)))

    return data_sampled


def pivot_gen(data, pivot_file):
    # Tworzenie pivot table
    pivot_table = data.pivot_table(index='title', columns='user_id', values='hours', aggfunc='sum')
    pivot_table.fillna(0, inplace=True)

    # Zapisanie do pliku pickle
    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', pivot_file)
    with open(pickle_file, 'wb') as f:
        pickle.dump(pivot_table, f)

    return pivot_table
