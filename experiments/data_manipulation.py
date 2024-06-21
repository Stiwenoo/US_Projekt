import pandas as pd
import os
import pickle


# Odczytanie pliku
# rec_name = 'recommendations.pkl'
# with open(os.path.join(os.path.dirname(__file__), '..', 'data', rec_name), 'rb') as file:
#     rec_data = pickle.load(file)
#
# games_name = 'games.pkl'
# with open(os.path.join(os.path.dirname(__file__), '..', 'data', games_name), 'rb') as file:
#     games_data = pickle.load(file)

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

# Wybieranie próbki zawierającej 100 tysięcy unikalnych użytkowników
sample_data = merged_data_unique.sample(n=100000, random_state=1)

# Tworzenie pivot table
pivot_table = sample_data.pivot_table(index='title', columns='user_id', values='hours', aggfunc='sum')
pivot_table.fillna(0, inplace=True)

# Zapisanie do pliku pickle
pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pivot_rec_games_100k.pkl')
with open(pickle_file, 'wb') as f:
    pickle.dump(pivot_table, f)
