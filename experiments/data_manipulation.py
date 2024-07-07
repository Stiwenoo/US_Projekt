import pandas as pd
import os
import pickle
import re
from nltk.stem import PorterStemmer


# Tworzenie dataframe'u dla modelu ALS
def dataframe_gen():
    games_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'games.csv'))
    rec_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'recommendations.csv'))

    # Usunięcie kolumn
    columns_to_drop_r = ['helpful', 'funny', 'is_recommended', 'review_id']
    rec_data.drop(columns=columns_to_drop_r, inplace=True)

    # Usunięcie kolumn
    columns_to_drop_g = ['date_release', 'win', 'mac', 'linux', 'rating', 'positive_ratio', 'user_reviews',
                         'price_final', 'price_original', 'discount', 'steam_deck']
    games_data.drop(columns=columns_to_drop_g, inplace=True)

    # Mergowanie
    merged_data = pd.merge(rec_data, games_data, on='app_id', how='inner')
    print(len(merged_data))
    print(merged_data.head())

    merged_data_unique = merged_data.drop_duplicates(subset=['app_id', 'user_id'])
    print(len(merged_data_unique))
    print(merged_data_unique.head())

    merged_data_unique['hours'] = merged_data_unique['hours'].astype('int32')

    # Zapisanie do pliku pickle
    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'rec_games.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(merged_data_unique, f)

    return merged_data_unique


# Próbkowanie danych z dataframe'u ALS
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


# Tworzenie tablicy przestawnej dla modelu ALS
def pivot_gen(data, pivot_file):
    # Tworzenie pivot table
    pivot_table = data.pivot_table(index='title', columns='user_id', values='hours', aggfunc='sum')
    pivot_table.fillna(0, inplace=True)

    # Zapisanie do pliku pickle
    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', pivot_file)
    with open(pickle_file, 'wb') as f:
        pickle.dump(pivot_table, f)

    return pivot_table


# Funkcja do usuwania HTML-owych tagów
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


#Funkcja usuwająca spacje z wyrazów
def remove_space(word):
    result = []
    for i in word:
        result.append(i. replace(' ', ''))
    return result


# Funkcja do połączenia wartości w jedną listę i zrobienia join ze spacją (Model CBR)
def join_columns(row):
    combined_list = []

    # Dodaj zawartość kolumn do listy, zamieniając stringi w listy
    combined_list.append(row['name'])
    combined_list.extend(row['categories'])
    combined_list.append(row['developer'])
    combined_list.append(row['publisher'])
    combined_list.extend(row['genres'])
    # combined_list.extend(row['short_description'])
    combined_list.extend(row['steamspy_tags'])
    combined_list.append(row['date_category'])

    # Konwertowanie wszystkich elementów listy na stringi
    combined_list = [str(item) for item in combined_list]

    # Zrób join ze spacją
    combined_string = ' '.join(combined_list)
    return combined_string


# Funkcja do klasyfikacji dat (Model CBR)
def categorize_date(date):
    if date < pd.Timestamp('2007-01-01'):
        return 'retrogry'
    elif date < pd.Timestamp('2016-01-01'):
        return 'starocie'
    else:
        return 'nowinki'


# Tworzenie dataframe'u dla modelu CBR
def dataframe_gen2(data_file):
    # with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'rec_games.pkl'), 'rb') as file:
    #     data = pickle.load(file)

    steam_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'steam.csv'))
    columns_to_drop_g = ['english', 'platforms', 'required_age', 'achievements',
                         'positive_ratings', 'negative_ratings', 'average_playtime',
                         'median_playtime', 'owners', 'price']
    steam_data.drop(columns=columns_to_drop_g, inplace=True)
    steam_data = steam_data.rename(columns={'appid': 'app_id'})

    steam_desc_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'steam_description_data.csv'))
    steam_desc_data = steam_desc_data.rename(columns={'steam_appid': 'app_id'})

    # Usunięcie HTML-owych tagów z kolumny 'short_description'
    steam_desc_data['short_description'] = steam_desc_data['short_description'].apply(remove_html_tags)

    # Usunięcie kolumn
    columns_to_drop_d = ['about_the_game']
    steam_desc_data.drop(columns=columns_to_drop_d, inplace=True)

    merged_data = pd.merge(steam_data, steam_desc_data, on='app_id', how='inner')
    merged_data['short_description'] = merged_data['short_description'].apply(lambda x: x. split())

    merged_data['release_date'] = pd.to_datetime(merged_data['release_date'])

    merged_data['categories'] = merged_data['categories'].apply(lambda x: x.split(';'))
    merged_data['categories'] = merged_data['categories'].apply(remove_space)
    merged_data['developer'] = merged_data['developer'].str.replace(' ', '')
    merged_data['publisher'] = merged_data['publisher'].str.replace(' ', '')
    merged_data['genres'] = merged_data['genres'].apply(lambda x: x.split(';'))
    merged_data['genres'] = merged_data['genres'].apply(remove_space)
    merged_data['steamspy_tags'] = merged_data['steamspy_tags'].apply(lambda x: x.split(';'))
    merged_data['steamspy_tags'] = merged_data['steamspy_tags'].apply(remove_space)
    # Tworzenie nowej kolumny z kategoriami
    merged_data['date_category'] = merged_data['release_date'].apply(categorize_date)

    # Zastosowanie funkcji do każdego wiersza w DataFrame
    merged_data['combined'] = merged_data.apply(join_columns, axis=1)

    # Zamiana wszystkich liter na małe w kolumnie 'combined'
    merged_data['combined'] = merged_data['combined'].str.lower()

    # Usuwa wszystkie przecinki i kropki
    merged_data['combined'] = merged_data['combined'].str.replace(',', '')
    merged_data['combined'] = merged_data['combined'].str.replace('.', '')

    # Usunięcie kolumn
    columns_to_drop_m = ['developer', 'publisher', 'categories', 'genres', 'detailed_description', 'short_description']
    merged_data.drop(columns=columns_to_drop_m, inplace=True)

    # Standaryzowanie slow
    ps = PorterStemmer()

    def stems(text):
        result = []
        for i in text.split():
            result.append(ps.stem(i))

        return " ".join(result)

    merged_data['combined'] = merged_data['combined'].apply(stems)

    # Zapisanie do pliku pickle
    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', data_file)
    with open(pickle_file, 'wb') as f:
        pickle.dump(merged_data, f)
