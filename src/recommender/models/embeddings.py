"""
Embeddings secuenciales (p.ej., Word2Vec sobre rutas):
- Entrenar embedding de POIs con secuencias de visits.
- Obtener vecinos similares al historial del usuario o al POI actual.
"""


def train_embeddings(sequences):
    # TODO: entrenar Word2Vec/Skip-gram sobre secuencias de POIs.
    raise NotImplementedError


def similar_pois(model, poi_id, topn=20):
    # TODO: vecinos en el espacio de embedding.
    raise NotImplementedError
