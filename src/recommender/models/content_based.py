"""
Content-based:
- TF-IDF de categorías (poi_categories.name) y/o primary_category.
- Perfil de usuario = media de vectores de POIs visitados.
- Similaridad coseno para puntuar candidatos.
- Opcional: ponderar por rating, total_ratings, price_tier/is_free (pendiente).
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix


def build_user_profile(user_items: Iterable[str], fsq_ids: List[str], tfidf_matrix: csr_matrix) -> np.ndarray:
    """Media de vectores TF-IDF de los POIs visitados."""
    seen = set(user_items)
    idxs = [i for i, fid in enumerate(fsq_ids) if fid in seen]
    if not idxs:
        return np.zeros((tfidf_matrix.shape[1],), dtype=np.float32)
    sub = tfidf_matrix[idxs]
    profile = sub.mean(axis=0)
    return np.asarray(profile).ravel()


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
