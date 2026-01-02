"""
Construcción de matrices de transición (Markov) para POI→POI y categoría→categoría.
- Entrada: visits_df (trail_id, venue_id, timestamp), poi_df (fsq_id, primary_category).
- Salida: dos diccionarios: transitions_poi[from_id][to_id] = prob, transitions_cat[from_cat][to_cat] = prob.
"""

from collections import Counter, defaultdict
from typing import Dict, Tuple

import pandas as pd


def _normalize(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def build_transitions(visits_df: pd.DataFrame, poi_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    if visits_df.empty or poi_df.empty:
        raise ValueError("visits_df o poi_df está vacío.")

    # Map fsq_id -> primary_category
    cat_map = poi_df.set_index("fsq_id")["primary_category"].to_dict()

    trans_poi_counts = defaultdict(Counter)
    trans_cat_counts = defaultdict(Counter)

    # Ordenar por trail_id y timestamp para secuencias
    visits_df = visits_df.sort_values(["trail_id", "timestamp"])

    for _, group in visits_df.groupby("trail_id"):
        seq = group["venue_id"].astype(str).tolist()
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            trans_poi_counts[a][b] += 1
            ca, cb = cat_map.get(a), cat_map.get(b)
            if ca and cb:
                trans_cat_counts[ca][cb] += 1

    # Normalizar a probabilidades
    trans_poi = {k: _normalize(v) for k, v in trans_poi_counts.items()}
    trans_cat = {k: _normalize(v) for k, v in trans_cat_counts.items()}
    return trans_poi, trans_cat


__all__ = ["build_transitions"]
