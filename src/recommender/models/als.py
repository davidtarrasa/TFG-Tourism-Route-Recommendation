"""ALS implícito para recomendaciones usuario-POI."""

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp


def build_interactions(visits) -> Tuple[sp.csr_matrix, Dict[int, int], Dict[str, int], List[str]]:
    """Construye matriz usuario-item (implícita) y mappings."""
    users = visits["user_id"].astype(int).unique()
    items = visits["venue_id"].astype(str).unique()
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {p: i for i, p in enumerate(items)}
    idx_to_item = list(items)

    rows = visits["user_id"].map(user_to_idx).to_numpy()
    cols = visits["venue_id"].map(item_to_idx).to_numpy()
    data = np.ones_like(rows, dtype=np.float32)
    mat = sp.coo_matrix((data, (rows, cols)), shape=(len(users), len(items))).tocsr()
    return mat, user_to_idx, item_to_idx, idx_to_item


def train_als(interactions: sp.csr_matrix, factors: int = 64, regularization: float = 0.01, iterations: int = 15, alpha: float = 40.0):
    """Entrena ALS implícito (requiere librería implicit)."""
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError as exc:
        raise ImportError("Falta la librería 'implicit'. Instálala para usar ALS.") from exc

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        calculate_training_loss=False,
    )
    model.fit(interactions * alpha)
    return model


def score_als(
    model,
    user_id: int,
    user_items: List[str],
    user_to_idx: Dict[int, int],
    item_to_idx: Dict[str, int],
    idx_to_item: List[str],
    topn: int = 500,
) -> Dict[str, float]:
    """Devuelve puntajes ALS para un usuario conocido o cold-start con historial."""
    seen = set(user_items)

    # User vector
    if user_id in user_to_idx:
        uvec = model.user_factors[user_to_idx[user_id]]
    else:
        idxs = [item_to_idx[x] for x in user_items if x in item_to_idx]
        if not idxs:
            return {}
        uvec = model.item_factors[idxs].mean(axis=0)

    scores_vec = np.dot(model.item_factors, uvec)

    # Tomar los mejores topn*3 para filtrar vistos
    cand_idx = np.argpartition(-scores_vec, min(len(scores_vec) - 1, topn * 3))[: topn * 3]
    scored = []
    for idx in cand_idx:
        fid = idx_to_item[idx]
        if fid in seen:
            continue
        scored.append((fid, float(scores_vec[idx])))
    scored.sort(key=lambda x: x[1], reverse=True)
    return dict(scored[:topn])


__all__ = ["build_interactions", "train_als", "score_als"]
