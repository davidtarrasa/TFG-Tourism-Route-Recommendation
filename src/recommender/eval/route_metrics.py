"""
Route-level metrics for evaluating itinerary quality.

This complements classic next-POI metrics (Hit/MRR/NDCG) by measuring:
- spatial coherence (legs not too short/too long, total distance)
- diversity (unique categories, entropy)
- alignment with a user profile (category match ratio)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..route_builder import haversine_km


def leg_distances_km(df: pd.DataFrame, anchor: Optional[Tuple[float, float]] = None) -> List[float]:
    """Distances between consecutive points, optionally from anchor to first POI."""
    if df.empty:
        return []
    coords = df[["lat", "lon"]].astype(float).to_numpy()
    dists: List[float] = []
    if anchor is not None:
        dists.append(float(haversine_km(anchor[0], anchor[1], coords[0, 0], coords[0, 1])))
    for i in range(len(coords) - 1):
        dists.append(float(haversine_km(coords[i, 0], coords[i, 1], coords[i + 1, 0], coords[i + 1, 1])))
    return dists


def shannon_entropy(items: Sequence[str]) -> float:
    if not items:
        return 0.0
    counts: Dict[str, int] = {}
    for x in items:
        if not x:
            continue
        counts[x] = counts.get(x, 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * log(p + 1e-12, 2)
    return float(ent)


def unique_ratio(items: Sequence[str]) -> float:
    if not items:
        return 0.0
    return float(len(set(items)) / max(1, len(items)))


def category_match_ratio(route_categories: Sequence[str], preferred_categories: Iterable[str]) -> float:
    pref = {str(x).lower() for x in preferred_categories if x}
    if not route_categories:
        return 0.0
    if not pref:
        return 0.0
    matches = sum(1 for c in route_categories if str(c).lower() in pref)
    return float(matches / len(route_categories))


@dataclass
class RouteMetrics:
    n_pois: int
    total_km: float
    avg_leg_km: float
    min_leg_km: float
    max_leg_km: float
    pct_legs_too_close: float
    pct_legs_too_far: float
    unique_cat_ratio: float
    cat_entropy: float
    cat_match_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_pois": float(self.n_pois),
            "total_km": float(self.total_km),
            "avg_leg_km": float(self.avg_leg_km),
            "min_leg_km": float(self.min_leg_km),
            "max_leg_km": float(self.max_leg_km),
            "pct_legs_too_close": float(self.pct_legs_too_close),
            "pct_legs_too_far": float(self.pct_legs_too_far),
            "unique_cat_ratio": float(self.unique_cat_ratio),
            "cat_entropy": float(self.cat_entropy),
            "cat_match_ratio": float(self.cat_match_ratio),
        }


def compute_route_metrics(
    ordered_df: pd.DataFrame,
    total_km: float,
    anchor: Optional[Tuple[float, float]],
    preferred_categories: Sequence[str],
    min_leg_km: float,
    max_leg_km: float,
) -> RouteMetrics:
    n = int(len(ordered_df))
    legs = leg_distances_km(ordered_df, anchor=anchor)
    avg_leg = float(np.mean(legs)) if legs else 0.0
    min_leg = float(np.min(legs)) if legs else 0.0
    max_leg = float(np.max(legs)) if legs else 0.0
    too_close = sum(1 for d in legs if d < min_leg_km)
    too_far = sum(1 for d in legs if d > max_leg_km)
    denom = max(1, len(legs))
    pct_close = float(too_close / denom)
    pct_far = float(too_far / denom)

    cats = ordered_df.get("primary_category")
    cat_list = [str(x) for x in cats.tolist()] if cats is not None else []
    u_ratio = unique_ratio(cat_list)
    ent = shannon_entropy([c.lower() for c in cat_list if c])
    match = category_match_ratio(cat_list, preferred_categories)

    return RouteMetrics(
        n_pois=n,
        total_km=float(total_km),
        avg_leg_km=avg_leg,
        min_leg_km=min_leg,
        max_leg_km=max_leg,
        pct_legs_too_close=pct_close,
        pct_legs_too_far=pct_far,
        unique_cat_ratio=u_ratio,
        cat_entropy=ent,
        cat_match_ratio=match,
    )


__all__ = [
    "RouteMetrics",
    "compute_route_metrics",
    "leg_distances_km",
    "shannon_entropy",
    "unique_ratio",
    "category_match_ratio",
]

