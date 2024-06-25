import implicit
import os
from scipy.sparse import csr_matrix
import pickle


def matrix_gen(pivot_file, matrix_file):
    # Tworzenie macierzy csr
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', pivot_file), 'rb') as file:
        pivot_table = pickle.load(file)

    sparse_matrix = csr_matrix(pivot_table)
    print('Macierz CSR przed transpozycja:\n', sparse_matrix)
    sparse_matrix = sparse_matrix.T.tocsr()
    print('Macierz CSR po transpozycji:\n', sparse_matrix)

    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'data', matrix_file)
    with open(pickle_file, 'wb') as f:
        pickle.dump(sparse_matrix, f)

    return sparse_matrix


def matrix_load(matrix_file):
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', matrix_file), 'rb') as file:
        sparse_matrix = pickle.load(file)

    return sparse_matrix


def model_gen(matrix_file, model_file):
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
    model.fit(matrix_load(matrix_file))

    pickle_file = os.path.join(os.path.dirname(__file__), '..', 'models', model_file)
    with open(pickle_file, 'wb') as f:
        pickle.dump(model, f)

    return model


def model_load(model_file):
    with open(os.path.join(os.path.dirname(__file__), '..', 'models', model_file), 'rb') as file:
        model = pickle.load(file)

    return model


# matrix_gen('pivot_9k_gamers.pkl', 'matrix_9k_gamers.pkl')
# model_gen('matrix_9k_gamers.pkl', 'als_model_9k_gamers.pkl')
