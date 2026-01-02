"""
Preparación de secuencias para entrenar embeddings (p.ej. Word2Vec) sobre rutas.
- Entrada: visits_df con columnas [trail_id, venue_id, timestamp].
- Salida: lista de secuencias (listas de fsq_id) ordenadas por trail_id y timestamp.
"""

from typing import List

import pandas as pd


def sequences_from_visits(visits_df: pd.DataFrame, min_len: int = 2) -> List[List[str]]:
    if visits_df.empty:
        return []
    seqs: List[List[str]] = []
    visits_df = visits_df.sort_values(["trail_id", "timestamp"])
    for _, group in visits_df.groupby("trail_id"):
        seq = group["venue_id"].astype(str).tolist()
        if len(seq) >= min_len:
            seqs.append(seq)
    return seqs


__all__ = ["sequences_from_visits"]
