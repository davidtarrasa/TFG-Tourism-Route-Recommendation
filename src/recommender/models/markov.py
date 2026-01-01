"""
Modelos secuenciales:
- Matriz de transiciones POI→POI y categoría→categoría.
- Dado current_poi o current_category, sugerir siguientes con mayor probabilidad.
- Empates: romper por distancia, rating, etc.
"""


def build_transition_matrices():
    # TODO: construir transiciones a partir de visits (ordenadas por trail/timestamp).
    raise NotImplementedError


def next_poi(current_poi, transitions_poi):
    # TODO: devolver candidatos ordenados por probabilidad.
    raise NotImplementedError


def next_category(current_category, transitions_cat):
    # TODO: devolver candidatos de categorías.
    raise NotImplementedError
