import implicit
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
import sys
import pickle

# rec_games_file = 'rec_games.pkl'
# with open(os.path.join(os.path.dirname(__file__), '..', 'data', rec_games_file), 'rb') as file:  
#     rec_games = pickle.load(file)

# print(rec_games)
# print(rec_games['hours'].dtypes)

# Wybieranie próbki zawierającej 80 tysięcy unikalnych użytkowników
# sample_data = rec_games.sample(n=80000, random_state=1)

# Tworzenie pivot table
# pivot_table = sample_data.pivot_table(index='title', columns='user_id', values='hours', aggfunc='sum')
# pivot_table.fillna(0, inplace=True)

# print(pivot_table.head())

# pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pivot_80k.pkl')
# with open(pickle_file, 'wb') as f:
#     pickle.dump(pivot_table, f)

pivot_80k_file = 'pivot_80k.pkl'
with open(os.path.join(os.path.dirname(__file__), '..', 'data', pivot_80k_file), 'rb') as file:  
    pivot_table = pickle.load(file)

print(pivot_table.index[4])
print(pivot_table.columns[69488])

# sparse_matrix = csr_matrix(pivot_table)
# print(sparse_matrix)

# pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'matrix_80k.pkl')
# with open(pickle_file, 'wb') as f:
#     pickle.dump(sparse_matrix, f)

matrix_80k_file = 'matrix_80k.pkl'
with open(os.path.join(os.path.dirname(__file__), '..', 'data', matrix_80k_file), 'rb') as file:  
    sparse_matrix = pickle.load(file)

print(sparse_matrix)

# model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

# model.fit(sparse_matrix)

# pickle_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'als_model_80k.pkl')
# with open(pickle_file, 'wb') as f:
#     pickle.dump(model, f)

als_model_80k_file = 'als_model_80k.pkl'
with open(os.path.join(os.path.dirname(__file__), '..', 'models', als_model_80k_file), 'rb') as file:  
    model = pickle.load(file)

ids, scores= model.similar_items(4, N=10)

print(ids)