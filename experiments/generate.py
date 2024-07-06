from exp_cbr import model_gen
import os
import sys

# Dodaj katalog nadrzędny do sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Ścieżki do plików
data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'rec_games_more.pkl')
model_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'cbr_model.pkl')

# Generowanie modelu
model_gen(data_file, model_file)
