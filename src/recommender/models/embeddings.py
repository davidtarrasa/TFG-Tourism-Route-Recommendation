"""
Embeddings secuenciales (p.ej., Word2Vec sobre rutas):
- Entrena embedding de POIs con secuencias de visits.
- Obtiene vecinos similares al historial del usuario o al POI actual.
"""

from typing import List, Tuple, Dict


def train_embeddings(
    sequences: List[List[str]],
    vector_size: int = 64,
    window: int = 10,
    min_count: int = 2,
    workers: int = 2,
):
    """Entrena un modelo Word2Vec (Skip-gram)."""
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError("Falta gensim; instálalo para usar embeddings") from exc

    if not sequences:
        raise ValueError("No hay secuencias para entrenar embeddings.")
    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        workers=workers,
    )
    return model


def similar_pois(model, poi_id: str, topn: int = 20) -> List[Tuple[str, float]]:
    """Vecinos en el espacio de embedding."""
    if poi_id not in model.wv:
        return []
    return model.wv.most_similar(poi_id, topn=topn)


def score_embeddings(model, user_items: List[str], topn: int = 20) -> Dict[str, float]:
    """Acumula similitudes de vecinos embedding para los POIs vistos."""
    from collections import Counter

    seen = set(user_items)
    scores = Counter()
    for fid in seen:
        if fid not in model.wv:
            continue
        for pid, sim in model.wv.most_similar(fid, topn=topn):
            if pid in seen:
                continue
            scores[pid] += sim
    return dict(scores)


__all__ = ["train_embeddings", "similar_pois", "score_embeddings"]
