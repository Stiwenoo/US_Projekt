import implicit
import os
from scipy.sparse import csr_matrix
import pickle


# Tworzenie macierzy csr
def matrix_gen(pivot_file, matrix_file):
    # pivot_file = 'pivot_rec_games_100k.pkl'
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', pivot_file), 'rb') as file:
        pivot_table = pickle.load(file)
    print(pivot_table)

    sparse_matrix = csr_matrix(pivot_table)
    print(sparse_matrix)

    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', matrix_file)
    with open(pickle_file, 'wb') as f:
        pickle.dump(sparse_matrix, f)


def matrix_load(matrix_file):
    # matrix_file = 'matrix_rec_games_100k.pkl'
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', matrix_file), 'rb') as file:
        sparse_matrix = pickle.load(file)
    return sparse_matrix


def model_gen(matrix_file, model_file):
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
    model.fit(matrix_load(matrix_file))

    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'models', model_file)
    with open(pickle_file, 'wb') as f:
        pickle.dump(model, f)


def model_load(model_file):
    with open(os.path.join(os.path.dirname(__file__), '..', 'models', model_file), 'rb') as file:
        model = pickle.load(file)
    return model


model_gen('matrix_rec_games_100k.pkl', 'als_model_100k.pkl')
model = model_load('als_model_100k.pkl')
ids, scores = model.similar_items(16685, N=100)
print(ids, scores)
