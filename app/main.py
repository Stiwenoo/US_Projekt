import implicit
import pandas as pd
import os
from scipy.sparse import csr_matrix

# Odczytywanie danych z pliku CSV
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'recommendations.csv'))
print("Odczytywanie danych z pliku CSV")

# Tworzenie mapowania z wartościami app_id i user_id na indeksy w macierzy
app_id_map = {app_id: idx for idx, app_id in enumerate(data['app_id'].unique())}
user_id_map = {user_id: idx for idx, user_id in enumerate(data['user_id'].unique())}
print("Tworzenie mapowania")

# Zamiana app_id i user_id na indeksy
data['app_idx'] = data['app_id'].map(app_id_map)
data['user_idx'] = data['user_id'].map(user_id_map)
print("Zamiana na indeksy")

# Tworzenie macierzy rzadkiej
sparse_matrix = csr_matrix((data['hours'], (data['app_idx'], data['user_idx'])), 
                            shape=(len(app_id_map), len(user_id_map)))

# Wyświetlenie wyniku
print(sparse_matrix.toarray())


# # Usunięcie duplikatów na podstawie kolumn 'app_id' i 'user_id'
# data_unique = data.drop_duplicates(subset=['app_id', 'user_id'])
# print("Usuniecie duplikatow na podstawie kolumn 'app_id' i 'user_id'")

# # Tworzenie tabeli przestawnej
# pivot_table = data.pivot_table(index='app_id', columns='user_id', values='hours', fill_value=0)
# print("Tworzenie tabeli przestawnej")

# # Konwersja tabeli przestawnej na macierz rzadką
# sparse_matrix = csr_matrix(pivot_table)
# print("Konwersja tabeli przestawnej na macierz rzadka")

# # Wyświetlenie wyniku
# print(sparse_matrix.toarray())
