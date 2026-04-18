"""
Content-based:
- TF-IDF de categorías (poi_categories.name) y/o primary_category.
- Perfil de usuario = media de vectores de POIs visitados.
- Similaridad coseno para puntuar candidatos.
- Opcional: ponderar por rating, total_ratings, price_tier/is_free (pendiente).
"""

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix


def build_user_profile(user_items: Iterable[str], fsq_ids: List[str], tfidf_matrix: csr_matrix) -> np.ndarray:
    """Perfil TF-IDF del usuario ponderado por frecuencia de visitas.

    POIs visitados más veces tienen mayor peso en el perfil, lo que refleja
    preferencias más fuertes que una simple media uniforme.
    """
    item_freq = Counter(str(x) for x in user_items)
    if not item_freq:
        return np.zeros((tfidf_matrix.shape[1],), dtype=np.float32)

    # Construir lista de (índice_en_matriz, peso) para los POIs conocidos
    indexed = [(i, item_freq[fid]) for i, fid in enumerate(fsq_ids) if fid in item_freq]
    if not indexed:
        return np.zeros((tfidf_matrix.shape[1],), dtype=np.float32)

    idxs = [i for i, _ in indexed]
    weights = np.array([w for _, w in indexed], dtype=np.float32)
    weights /= weights.sum()

    sub = tfidf_matrix[idxs]
    profile = np.asarray(sub.multiply(weights.reshape(-1, 1)).sum(axis=0)).ravel()
    return profile.astype(np.float32)


def score_content(user_items: Iterable[str], fsq_ids: List[str], tfidf_matrix: csr_matrix) -> Dict[str, float]:
    """
    Calcula similitud coseno entre el perfil del usuario y todos los POIs.
    Devuelve dict fsq_id -> score (excluye ítems ya vistos).
    """
    seen = set(user_items)
    profile = build_user_profile(user_items, fsq_ids, tfidf_matrix)
    if profile.max() == 0:
        return {}

    # coseno = (p . M^T) / (||p|| * ||item||)
    item_norms = np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).A.ravel() + 1e-8
    profile_norm = np.linalg.norm(profile) + 1e-8
    sims = (tfidf_matrix @ profile) / (item_norms * profile_norm)

    scores = {}
    for fid, sim in zip(fsq_ids, sims):
        if fid in seen:
            continue
        scores[fid] = float(sim)
    return scores


__all__ = ["build_user_profile", "score_content"]
