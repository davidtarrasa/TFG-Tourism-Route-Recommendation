"""
Content-based:
- TF-IDF de categorías (poi_categories.name) y/o primary_category.
- Perfil de usuario = media de vectores de POIs visitados.
- Similaridad coseno para puntuar candidatos.
- Opcional: ponderar por rating, total_ratings, price_tier/is_free.
"""


def build_item_tfidf():
    # TODO: generar matriz TF-IDF de POIs.
    raise NotImplementedError


def score_content(user_profile, item_matrix):
    # TODO: calcular similitud y devolver scores.
    raise NotImplementedError
