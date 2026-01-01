"""
Co-visitas / item-item:
- Matriz de co-ocurrencia POI↔POI (misma ruta/usuario).
- Score = suma de similitudes con POIs del usuario.
"""


def build_co_visitation_matrix():
    # TODO: contar co-ocurrencias y normalizar.
    raise NotImplementedError


def score_co_visitation(user_items, co_matrix):
    # TODO: sumar similitudes para candidatos.
    raise NotImplementedError
