"""
Greedy route planner with simple constraints.

Why:
- Selecting top-K and then ordering with NN/2-opt can produce legs that are
  too short (redundant) or too long (jumps).
- We want category alignment *and* diversity, plus distance coherence.

This planner builds an ordered route step-by-step from an anchor.
It is intentionally heuristic (fast, controllable via config).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .route_builder import haversine_km


@dataclass
class PlannedRoute:
    ordered_df: pd.DataFrame
    total_km: float
    legs_km: List[float]


def _leg_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(haversine_km(a[0], a[1], b[0], b[1]))


def plan_route(
    candidates: pd.DataFrame,
    k: int,
    anchor: Optional[Tuple[float, float]] = None,
    min_leg_km: float = 0.3,
    max_leg_km: float = 5.0,
    pair_min_km: float = 0.2,
    max_per_category: int = 2,
    distance_weight: float = 0.35,
    diversity_bonus: float = 0.05,
) -> PlannedRoute:
    """
    Build an ordered route of length k from candidate POIs.

    candidates must contain: fsq_id, score, lat, lon, primary_category.
    """
    if candidates.empty or k <= 0:
        return PlannedRoute(candidates.head(0), 0.0, [])

    df = candidates.copy()
    df = df.dropna(subset=["lat", "lon", "score"], how="any")
    if df.empty:
        return PlannedRoute(df, 0.0, [])

    # Start point for legs: anchor if provided, otherwise first selected POI.
    selected_rows = []
    legs: List[float] = []
    cat_counts: Dict[str, int] = {}

    # Precompute coordinates for speed.
    coords = df[["lat", "lon"]].astype(float).to_numpy()
    ids = df["fsq_id"].astype(str).tolist()
    cats = df.get("primary_category")
    cat_list = [str(x) for x in cats.tolist()] if cats is not None else [""] * len(df)
    base_scores = df["score"].astype(float).to_numpy()

    selected_mask = np.zeros(len(df), dtype=bool)

    # If no anchor, pick the best-scoring POI as start.
    if anchor is None:
        start_idx = int(np.argmax(base_scores))
        selected_mask[start_idx] = True
        selected_rows.append(df.iloc[start_idx])
        last_point = (float(coords[start_idx, 0]), float(coords[start_idx, 1]))
        cat = cat_list[start_idx]
        if cat:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    else:
        last_point = (float(anchor[0]), float(anchor[1]))

    while len(selected_rows) < k:
        best_idx = None
        best_u = -1e18

        # Compute distances from last_point.
        dists = np.array([_leg_km(last_point, (float(coords[i, 0]), float(coords[i, 1]))) for i in range(len(df))], dtype=float)

        for i in range(len(df)):
            if selected_mask[i]:
                continue

            # Hard-ish constraints
            if dists[i] < pair_min_km:
                continue
            if len(selected_rows) > 0:
                # Only enforce leg constraints after we have at least 1 POI.
                if dists[i] < min_leg_km or dists[i] > max_leg_km:
                    continue

            cat = cat_list[i]
            if cat and cat_counts.get(cat, 0) >= max_per_category:
                continue

            # Utility: base score penalized by distance, plus diversity bonus
            u = float(base_scores[i]) - distance_weight * float(dists[i])
            if cat and cat_counts.get(cat, 0) == 0:
                u += diversity_bonus

            if u > best_u:
                best_u = u
                best_idx = i

        if best_idx is None:
            # Relax constraints if we get stuck: allow distances outside [min,max] but still avoid too-close.
            for i in range(len(df)):
                if selected_mask[i]:
                    continue
                if dists[i] < pair_min_km:
                    continue
                cat = cat_list[i]
                if cat and cat_counts.get(cat, 0) >= max_per_category:
                    continue
                u = float(base_scores[i]) - distance_weight * float(dists[i])
                if cat and cat_counts.get(cat, 0) == 0:
                    u += diversity_bonus
                if u > best_u:
                    best_u = u
                    best_idx = i

        if best_idx is None:
            break

        selected_mask[best_idx] = True
        selected_rows.append(df.iloc[best_idx])
        new_point = (float(coords[best_idx, 0]), float(coords[best_idx, 1]))
        legs.append(_leg_km(last_point, new_point))
        last_point = new_point
        cat = cat_list[best_idx]
        if cat:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    out_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    total_km = float(sum(legs))
    return PlannedRoute(out_df, total_km, legs)


__all__ = ["PlannedRoute", "plan_route"]

