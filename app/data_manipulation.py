import implicit
import pandas as pd
import os
from scipy.sparse import csr_matrix
import sys
import pickle


# Odczytanie pliku
rec_name = 'recommendations.pkl'
with open(os.path.join(os.path.dirname(__file__), '..', 'data', rec_name), 'rb') as file:  
    rec_data = pickle.load(file)

games_name = 'games.pkl'
with open(os.path.join(os.path.dirname(__file__), '..', 'data', games_name), 'rb') as file:  
    games_data = pickle.load(file)


print(len(rec_data))
print(len(games_data))

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

# games_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'games.csv'))

# Wyświetlenie początkowych wierszy danych
# print("Przed usunięciem kolumn:")
# print(rec_data.head())

# Usunięcie kolumn
# columns_to_drop_r = ['helpful', 'funny', 'is_recommended', 'review_id']
# rec_data.drop(columns=columns_to_drop_r, inplace=True)

# Usunięcie kolumn
# columns_to_drop_g = ['date_release','win','mac','linux','rating','positive_ratio','user_reviews','price_final','price_original','discount','steam_deck']
# games_data.drop(columns=columns_to_drop_g, inplace=True)

# Wyświetlenie danych po usunięciu kolumn
# print("\nPo usunięciu kolumn:")
# print(rec_data.head())

# print(type(rec_data['app_id']))

# Zapisanie do pliku pickle
# pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'games.pkl')
# with open(pickle_file, 'wb') as f:
#     pickle.dump(games_data, f)

# print(f"\nDane zapisane do pliku: {pickle_file}")
