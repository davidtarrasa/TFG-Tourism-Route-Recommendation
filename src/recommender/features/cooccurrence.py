"""
Construcción de co-ocurrencias POI↔POI a partir de secuencias (trail_id/usuario).
- Entrada: visits_df con columnas [trail_id, user_id, venue_id].
- Salida: (matriz co-ocurrencia sparse, mapping id->idx, mapping idx->id)
"""

from collections import Counter
from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def build_cooccurrence(visits_df: pd.DataFrame) -> Tuple[coo_matrix, Dict[str, int], Dict[int, str]]:
    """Cuenta co-ocurrencias de POIs por trail_id (pares únicos)."""
    if visits_df.empty:
        raise ValueError("visits_df está vacío.")

    # Mapear fsq_id a índices
    unique_ids = visits_df["venue_id"].astype(str).unique()
    id_to_idx = {vid: i for i, vid in enumerate(unique_ids)}
    idx_to_id = {i: vid for vid, i in id_to_idx.items()}

    co_counts = Counter()
    # Agrupar por trail_id (o user_id si se prefiere); aquí trail_id
    for _, group in visits_df.groupby("trail_id"):
        venues = set(group["venue_id"].astype(str))
        # Generar pares únicos sin repetir
        for a, b in combinations(sorted(venues), 2):
            co_counts[(a, b)] += 1
            co_counts[(b, a)] += 1  # simétrica

    if not co_counts:
        raise ValueError("No se encontraron co-ocurrencias.")

    rows, cols, data = [], [], []
    for (a, b), cnt in co_counts.items():
        rows.append(id_to_idx[a])
        cols.append(id_to_idx[b])
        data.append(cnt)

    mat = coo_matrix((data, (rows, cols)), shape=(len(unique_ids), len(unique_ids)), dtype=np.float32)
    return mat, id_to_idx, idx_to_id


__all__ = ["build_cooccurrence"]
