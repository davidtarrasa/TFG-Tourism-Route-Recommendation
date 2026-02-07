"""
Embeddings secuenciales (p.ej., Word2Vec sobre rutas):
- Entrena embedding de POIs con secuencias de visits.
- Obtiene vecinos similares al historial del usuario o al POI actual.
"""

from typing import List, Tuple, Dict


def train_embeddings(
    sequences: List[List[str]],
    vector_size: int = 128,
    window: int = 15,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 10,
    negative: int = 5,
    sample: float = 1e-3,
    ns_exponent: float = 0.75,
    hs: int = 0,
    seed: int = 42,
):
    """Entrena un modelo Word2Vec (Skip-gram) sobre secuencias de POIs."""
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
        epochs=epochs,
        negative=negative,
        sample=sample,
        ns_exponent=ns_exponent,
        hs=hs,
        seed=seed,
    )
    return model


def similar_pois(model, poi_id: str, topn: int = 20) -> List[Tuple[str, float]]:
    """Vecinos en el espacio de embedding."""
    if poi_id not in model.wv:
        return []
    return model.wv.most_similar(poi_id, topn=topn)


def score_embeddings_next(model, current_poi: str, topn: int = 50) -> Dict[str, float]:
    """Score "next POI" usando solo el POI actual (secuencial)."""
    if current_poi not in model.wv:
        return {}
    out: Dict[str, float] = {}
    for pid, sim in model.wv.most_similar(current_poi, topn=topn):
        if pid == current_poi:
            continue
        out[pid] = float(sim)
    return out


def score_embeddings_context(model, context_items: List[str], topn: int = 50) -> Dict[str, float]:
    """
    Score "next POI" usando un contexto (promedio de embeddings de los Ãºltimos N POIs).

    Gensim permite pasar una lista de tokens "positive" y promedia los vectores internamente.
    """
    if not context_items:
        return {}
    ctx = [str(x) for x in context_items if x and str(x) in model.wv]
    if not ctx:
        return {}
    out: Dict[str, float] = {}
    for pid, sim in model.wv.most_similar(positive=ctx, topn=topn):
        if pid in ctx:
            continue
        out[pid] = float(sim)
    return out


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


__all__ = ["train_embeddings", "similar_pois", "score_embeddings_next", "score_embeddings_context", "score_embeddings"]
