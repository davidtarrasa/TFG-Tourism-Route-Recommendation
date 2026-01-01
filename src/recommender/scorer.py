"""
Scorer/orquestador de recomendación.
- Combina puntajes de motores (content, co-visitas, markov, embeddings/ALS opcional).
- Aplica filtros (ciudad, categorías, price_tier/is_free) y re-ranking por distancia.
- Devuelve top-K y, opcionalmente, un orden para ruta.
"""


def score_candidates():
    # TODO: implementar combinación de scores y re-ranking.
    raise NotImplementedError
