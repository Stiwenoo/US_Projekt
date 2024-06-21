import implicit
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
import sys
import pickle

pivot_name = 'recommendations_pivot.pkl'
with open(os.path.join(os.path.dirname(__file__), '..', 'data', pivot_name), 'rb') as file:  
    pivot_table = pickle.load(file)

print(pivot_table.index[-1])

app_id_index_mapping = {num: index for num, index in enumerate(pivot_table.index)}
# print(app_id_index_mapping)

matrix_name = 'recommendations_matrix.pkl'
with open(os.path.join(os.path.dirname(__file__), '..', 'data', matrix_name), 'rb') as file:  
    sparse_matrix = pickle.load(file)

print(sparse_matrix.row_ind[-1])

# Wyświetlenie pierwszych pięciu wierszy macierzy wraz z zamienionymi app_id
# print("Pierwsze pięć wierszy macierzy CSR z zamienionymi app_id:")
# for i in range(min(5, sparse_matrix.shape[0])):
#     print("jeszcze git: "+str(i))
#     original_app_id = pivot_table.loc[i, pivot_table.index]
#     mapped_app_id = app_id_index_mapping[original_app_id]
#     print(f"app_id={original_app_id} (zmapowane na {mapped_app_id}):", sparse_matrix.getrow(i).toarray())



# Zapisanie do pliku pickle
# matrix_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'recommendations_matrix.pkl')
# with open(matrix_file, 'wb') as f:
#     pickle.dump(sparse_matrix, f)
