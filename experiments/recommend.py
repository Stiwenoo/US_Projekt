import pickle
import os
import pandas as pd
import numpy as np
import implicit

model_name = 'als_model_100k.pkl'
with open(os.path.join(os.path.dirname(__file__), '..', 'models', model_name), 'rb') as file:  
    model = pickle.load(file)

ids, scores = model.similar_items(16685, N=30)
print(ids, scores)

print(type(ids.tolist()))
print(type(scores.tolist()))


# Wybór przykładowej gry (app_id) do znalezienia podobnych gier
# game_id = 10  # Zmienna przykładowa, można zmienić na dowolne app_id

# Generowanie podobnych gier do wybranej gry
# app_ids, scores = model.similar_items(game_id, N=100)  # N oznacza liczbę podobnych gier do znalezienia
# print(similar_games)

# obj_type = type(similar_games).__name__
# if isinstance(similar_games, list):
#     print(f'Typ obiektu similar_games: list')
#     for idx, game in enumerate(similar_games):
#         print(f'Element {idx + 1}: {game}, Typ: {type(game)}')
# if isinstance(similar_games, tuple):
#     print(f'Typ obiektu similar_games: tuple')
#     for idx, game in enumerate(similar_games):
#         print(f'Element {idx + 1}: {game}, Typ: {type(game).__name__}')
# else:
#     print(f'Typ obiektu similar_games: {obj_type}')
# app_ids = similar_games[0].astype(int).tolist()
# print(app_ids)

# Przetworzenie similar_games z ndarray na odpowiednią strukturę
# app_ids, scores = similar_games
# app_ids = similar_games[0].astype(int).tolist()  # Konwersja na listę Pythona
# scores = similar_games[1].tolist()  # Konwersja na listę Pythona


# print(app_ids)
# print(scores)

# Mapowanie app_id na tytuły gier tylko z similar_games
# app_id_to_title = dict(zip(games_data['app_id'], games_data['title']))
# print(app_id_to_title)
# print(app_id_to_title[0])
# print(type(app_id_to_title[10]))


def mapping_game_titles(pivot_name):
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', pivot_name), 'rb') as file:
        pivot_table = pickle.load(file)

    # Tworzymy słownik mapujący indeksy z pivot_table do macierzy rzadkiej
    index_to_app_id = {i: app_id for i, app_id in enumerate(pivot_table.index)}
    return index_to_app_id


# print(mapping_game_titles('pivot_rec_games_100k.pkl'))
# Sprawdzenie, czy game_id jest w indeksie pivot_table
# game_id = 10  # Przykładowe app_id, można zmienić na dowolne
# if game_id not in pivot_table.index:
#     raise ValueError(f"game_id {game_id} is not in pivot_table index")

# Generujemy rekomendacje
# game_index = pivot_table.index.get_loc(game_id)  # Pobranie indeksu dla game_id
# similar_games = model.similar_items(game_index, N=20)  # Pobieramy więcej niż potrzebne



# Wyciągamy app_ids i mapujemy na oryginalne app_id
# app_ids = [index_to_app_id[i] for i in similar_games[0].astype(int) if i in index_to_app_id]

# Przycinamy listę do 10 elementów, jeśli jest to konieczne
# app_ids = app_ids[:10]

# Wyświetlamy zmapowane app_ids
# print("Zmapowane app_ids:", app_ids)
# mapped_list = [app_id_to_title[value] for value in app_ids]
# print("Zmapowane tytuly:", mapped_list)

# app_id_index_mapping = {num: index for num, index in enumerate(pivot_table.index)}
# print(app_id_index_mapping)

# mapped_list = [app_id_index_mapping[value] for value in app_ids]
# print(mapped_list)
# print(app_id_index_mapping)

# Iterujemy przez indeksy obu list jednocześnie
# for i in range(len(app_ids)):
#     app_id = str(app_ids[i])
#     score = scores[i]
    
#     # Sprawdzamy, czy app_id istnieje w słowniku app_id_to_title
#     if app_id in app_id_to_title:
#         title = app_id_to_title[app_id]
#         print(f"{title} - {score}")
#     else:
#         print(f"Nieznany tytuł (app_id={app_id}) - {score}")
