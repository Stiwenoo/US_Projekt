import implicit
import pandas as pd
import os
from scipy.sparse import csr_matrix
import sys
import pickle

data_name = 'recommendations.pkl'
with open(os.path.join(os.path.dirname(__file__), '..', 'data', data_name), 'rb') as file:  
    data = pickle.load(file)

# Odczytywanie danych z pliku CSV
# data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'recommendations.csv'))
# print("Odczytywanie danych z pliku CSV")

# Sprawdzenie podstawowych informacji o danych
# print(f"Liczba wierszy w danych: {len(data)}")
# print(f"Liczba unikalnych gier (app_id): {data['app_id'].nunique()}")

# print(f"Liczba unikalnych uzytkownikow (user_id): {data['user_id'].nunique()}\n")

# Przekształcenie kolumny 'date' na format daty
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Filtrowanie danych do rekomendacji z roku 2020 i nowszych
data_filtered = data[data['date'] >= '2020-01-01']

# Zaokrąglenie wartości w kolumnie 'hours' do liczb całkowitych
data_filtered.loc[:, 'hours'] = data_filtered['hours'].round().astype('int64')

# Sprawdzenie podstawowych informacji o danych po filtrowaniu
# print(f"Liczba wierszy po filtrowaniu: {len(data_filtered)}")
# print(f"Liczba unikalnych gier (app_id) po filtrowaniu: {data_filtered['app_id'].nunique()}")
# print(f"Liczba unikalnych uzytkownikow (user_id) po filtrowaniu: {data_filtered['user_id'].nunique()}\n")

# Usunięcie duplikatów na podstawie kolumn 'app_id' i 'user_id'
data_unique = data_filtered.drop_duplicates(subset=['app_id', 'user_id'])
# print("Usuniecie duplikatow na podstawie kolumn 'app_id' i 'user_id'\n")

# Filtrowanie użytkowników (user_id) - na przykład wybór pierwszych 1000 unikalnych użytkowników
sampled_users = data_unique['user_id'].unique()[:30000]

# Filtrowanie danych do wybranych użytkowników
data_sampled = data_unique[data_unique['user_id'].isin(sampled_users)]
print(type(data_sampled))
print(data_sampled.head())

# Sprawdzenie podstawowych informacji o próbkowanych danych
# print(f"Liczba wierszy po probkowaniu: {len(data_sampled)}")
# print(f"Liczba unikalnych gier (app_id) po probkowaniu: {data_sampled['app_id'].nunique()}")
# print(f"Liczba unikalnych uzytkownikow (user_id) po probkowaniu: {data_sampled['user_id'].nunique()}\n")

# # Tworzenie tabeli przestawnej
print("Tworzenie tabeli przestawnej")
pivot_table = data_sampled.pivot_table(index='app_id', columns='user_id', values='hours', fill_value=0)

# Zapisanie do pliku pickle
pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'recommendations_pivot.pkl')
with open(pickle_file, 'wb') as f:
    pickle.dump(pivot_table, f)


# Konwersja tabeli przestawnej na macierz rzadką
sparse_matrix = csr_matrix(pivot_table.astype('int64'))
print("Konwersja tabeli przestawnej na macierz rzadka")

# Zapisanie do pliku pickle
matrix_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'recommendations_matrix.pkl')
with open(matrix_file, 'wb') as f:
    pickle.dump(sparse_matrix, f)

# Wyświetlenie wyniku
# print(sparse_matrix.toarray())

# Inicjalizacja modelu ALS
model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15)

# Wytrenowanie modelu za pomocą macierzy rzadkiej
model.fit(sparse_matrix)

model_pkl_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'als_model_30000k.pkl')
with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)
