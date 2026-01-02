"""
Co-visitas / item-item:
- Matriz de co-ocurrencia POI↔POI (misma ruta/usuario).
- Score = suma de pesos con POIs del usuario (simetría simple).
"""

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def to_csr(mat: coo_matrix) -> csr_matrix:
    """Convierte a CSR para operaciones rápidas."""
    return mat.tocsr()


def score_co_visitation(
    user_items: Iterable[str],
    co_matrix: coo_matrix | csr_matrix,
    id_to_idx: Dict[str, int],
    idx_to_id: Dict[int, str],
) -> Dict[str, float]:
    """
    Suma las co-ocurrencias de los POIs vistos del usuario hacia todos los demás.
    Excluye ítems ya vistos.
    """
    seen = set(user_items)
    csr = co_matrix if isinstance(co_matrix, csr_matrix) else co_matrix.tocsr()
    scores = np.zeros(csr.shape[0], dtype=np.float32)

    for fid in seen:
        idx = id_to_idx.get(fid)
        if idx is None:
            continue
        scores += csr[idx].toarray().ravel()

    results: Dict[str, float] = {}
    for i, val in enumerate(scores):
        fid = idx_to_id[i]
        if fid in seen:
            continue
        if val > 0:
            results[fid] = float(val)
    return results


__all__ = ["score_co_visitation", "to_csr"]
