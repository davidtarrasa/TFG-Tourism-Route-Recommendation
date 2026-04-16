"""
Multi-route contract service.

Generates up to 4 route variants in one request:
- history
- inputs
- location
- full

Rules:
- Missing signals do not affect unrelated variants.
- For new users (no history), at least one request input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .category_intents import INCONCLUSIVE, classify_category_intent, infer_user_intents
from .prefs import Prefs, normalize_categories
from .scorer import recommend
from .config import DEFAULT_CONFIG_PATH, load_config
from .utils_db import get_conn, load_pois


@dataclass
class MultiRouteResult:
    request_signals: Dict[str, bool]
    user_exists: bool
    routes: Dict[str, pd.DataFrame]
    omitted: Dict[str, str]
    warnings: List[str]


def _stable_seed_for_surprise(
    *,
    base_seed: int,
    city_qid: Optional[str],
    user_id: Optional[int],
    lat: Optional[float],
    lon: Optional[float],
    k: int,
) -> int:
    payload = f"{city_qid}|{user_id}|{lat}|{lon}|{k}".encode("utf-8", errors="ignore")
    h = hashlib.blake2b(payload, digest_size=8).digest()
    v = int.from_bytes(h, byteorder="little", signed=False)
    return int((int(base_seed) + v) % (2**32 - 1))


def _collect_row_intents(df: pd.DataFrame) -> set:
    intents = set()
    if df is None or df.empty:
        return intents
    if "primary_category" not in df.columns:
        return intents
    for c in df["primary_category"].astype(str).tolist():
        c = str(c).strip()
        if not c or c.lower() in {"nan", "none", "null"}:
            continue
        i, _, _ = classify_category_intent(c, use_semantic=False)
        if i and i != INCONCLUSIVE:
            intents.add(i)
    return intents


def _inject_soft_surprise(
    *,
    route_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    k: int,
    prefs: Optional[Prefs],
    lat: Optional[float],
    lon: Optional[float],
    city_qid: Optional[str],
    user_id: Optional[int],
) -> Tuple[pd.DataFrame, bool]:
    cfg = load_config(DEFAULT_CONFIG_PATH, city_qid=city_qid)
    s_cfg = cfg.get("surprise", {})
    if not bool(s_cfg.get("enabled", True)):
        return route_df, False
    rate = float(s_cfg.get("soft_rate", 0.03))
    if rate <= 0:
        return route_df, False
    if route_df is None or route_df.empty or pool_df is None or pool_df.empty:
        return route_df, False

    seed = _stable_seed_for_surprise(
        base_seed=int(s_cfg.get("seed", 42)),
        city_qid=city_qid,
        user_id=user_id,
        lat=lat,
        lon=lon,
        k=int(k),
    )
    rng = np.random.default_rng(seed)
    if float(rng.random()) >= rate:
        out = route_df.copy()
        out["is_surprise"] = False
        return out, False

    route = route_df.copy()
    route["is_surprise"] = False
    route_ids = set(route["fsq_id"].astype(str)) if "fsq_id" in route.columns else set()
    pool = pool_df.copy()
    if "fsq_id" not in pool.columns:
        return route, False
    pool = pool[~pool["fsq_id"].astype(str).isin(route_ids)].copy()
    if pool.empty:
        return route, False

    pool = _coalesce_latlon_columns(pool)
    pool["score"] = pd.to_numeric(pool.get("score", 0.0), errors="coerce").fillna(0.0)
    pool["lat"] = pd.to_numeric(pool.get("lat"), errors="coerce")
    pool["lon"] = pd.to_numeric(pool.get("lon"), errors="coerce")
    pool = pool.dropna(subset=["lat", "lon"], how="any")
    if pool.empty:
        return route, False

    # Keep "soft" candidates: decent score and not too far from anchor/route center.
    smax = float(pool["score"].max()) if not pool.empty else 0.0
    min_ratio = float(s_cfg.get("min_score_ratio", 0.35))
    if smax > 0:
        pool = pool[pool["score"] >= (min_ratio * smax)].copy()
    if pool.empty:
        return route, False

    if lat is not None and lon is not None:
        ref_lat, ref_lon = float(lat), float(lon)
    else:
        r2 = _coalesce_latlon_columns(route)
        if "lat" in r2.columns and "lon" in r2.columns:
            r2 = r2.dropna(subset=["lat", "lon"], how="any")
            if not r2.empty:
                ref_lat = float(pd.to_numeric(r2["lat"], errors="coerce").mean())
                ref_lon = float(pd.to_numeric(r2["lon"], errors="coerce").mean())
            else:
                ref_lat, ref_lon = None, None
        else:
            ref_lat, ref_lon = None, None

    if ref_lat is not None and ref_lon is not None:
        pool["__dist_ref"] = pool.apply(
            lambda r: _haversine_km(ref_lat, ref_lon, float(r["lat"]), float(r["lon"])),
            axis=1,
        )
        soft_max_km = float(s_cfg.get("soft_max_km", 3.0))
        pool = pool[pool["__dist_ref"] <= soft_max_km].copy()
    else:
        pool["__dist_ref"] = 0.0
    if pool.empty:
        return route, False

    pref_intents = set()
    if prefs and prefs.categories:
        pref_intents = set(i for i in infer_user_intents(normalize_categories(prefs.categories)) if i and i != INCONCLUSIVE)
    route_intents = _collect_row_intents(route)
    target_intents = pref_intents or route_intents

    def _cand_intent(row: pd.Series) -> str:
        c = str(row.get("primary_category") or "").strip()
        if not c:
            return INCONCLUSIVE
        i, _, _ = classify_category_intent(c, use_semantic=False)
        return i

    pool["__intent"] = pool.apply(_cand_intent, axis=1)
    pool["__intent_match"] = pool["__intent"].apply(lambda i: 1.0 if i in target_intents else 0.0)

    # Soft surprise ranking: still "reasonable" and close.
    pmin = float(pool["score"].min())
    pmax = float(pool["score"].max())
    if pmax - pmin <= 1e-12:
        pool["__s_norm"] = 0.0
    else:
        pool["__s_norm"] = (pool["score"] - pmin) / (pmax - pmin)
    dmax = float(pool["__dist_ref"].max()) if not pool.empty else 0.0
    if dmax <= 1e-12:
        pool["__d_pref"] = 1.0
    else:
        pool["__d_pref"] = 1.0 - (pool["__dist_ref"] / dmax)
    pool["__surprise_soft_score"] = 0.55 * pool["__s_norm"] + 0.30 * pool["__d_pref"] + 0.15 * pool["__intent_match"]
    pool = pool.sort_values("__surprise_soft_score", ascending=False).head(int(max(10, min(40, len(pool)))))
    if pool.empty:
        return route, False

    chosen = pool.iloc[int(rng.integers(0, len(pool)))]
    cand = chosen.to_dict()
    cand["is_surprise"] = True
    # Replace last slot to minimize route disruption.
    out = route.head(int(k)).copy()
    if out.empty:
        return route, False
    out.iloc[len(out) - 1] = pd.Series(cand).reindex(out.columns)
    if "fsq_id" in out.columns:
        out = out.drop_duplicates(subset=["fsq_id"], keep="first")
    if len(out) < int(k):
        filler = route[~route["fsq_id"].astype(str).isin(set(out["fsq_id"].astype(str)))].copy()
        if not filler.empty:
            filler["is_surprise"] = False
            out = pd.concat([out, filler], ignore_index=True).head(int(k))
    out = out.reset_index(drop=True)
    return out, True


def _has_inputs(prefs: Optional[Prefs], free_only: bool, max_price_tier: Optional[int]) -> bool:
    if prefs and prefs.categories:
        return True
    if prefs and prefs.free_only is not None:
        return True
    if prefs and prefs.max_price_tier is not None:
        return True
    if free_only:
        return True
    if max_price_tier is not None:
        return True
    return False


def _user_exists(dsn: Optional[str], user_id: Optional[int], city_qid: Optional[str]) -> bool:
    if user_id is None:
        return False
    try:
        conn = get_conn(dsn)
        q = "SELECT 1 FROM visits WHERE user_id = %(uid)s"
        params = {"uid": int(user_id)}
        if city_qid:
            q += " AND venue_city = %(city_qid)s"
            params["city_qid"] = city_qid
        q += " LIMIT 1"
        df = pd.read_sql(q, conn, params=params)
        return not df.empty
    except Exception:
        return False


def _trim_inputs_with_category_coverage(df: pd.DataFrame, prefs: Optional[Prefs], k: int) -> pd.DataFrame:
    """
    Ensure `inputs` route visibly reflects requested categories/intents.

    Rule:
    - If number of requested categories/intents <= k, try to include at least
      one POI per requested intent/category (when available), then fill by score.
    - If there are no usable preferences, fallback to top-k by score.
    """
    if df is None or df.empty:
        return df
    if not prefs or not prefs.categories:
        return df.head(k).reset_index(drop=True)

    pref_cats = normalize_categories(prefs.categories)
    if not pref_cats:
        return df.head(k).reset_index(drop=True)

    requested_intents = [i for i in infer_user_intents(pref_cats) if i and i != INCONCLUSIVE]
    requested_tokens = requested_intents if requested_intents else [c.lower() for c in pref_cats]

    # Respect product rule: only enforce one-per-category when request size <= k.
    if len(requested_tokens) > int(k):
        return df.head(k).reset_index(drop=True)

    ranked = df.copy()
    if "score" in ranked.columns:
        ranked = ranked.sort_values("score", ascending=False)

    # Precompute per-row signals.
    def _row_primary(row: pd.Series) -> str:
        return str(row.get("primary_category") or "").strip()

    def _row_intent(row: pd.Series) -> str:
        primary = _row_primary(row)
        if not primary:
            return INCONCLUSIVE
        intent, _, _ = classify_category_intent(primary, use_semantic=False)
        return intent

    ranked["__primary"] = ranked.apply(_row_primary, axis=1)
    ranked["__intent"] = ranked.apply(_row_intent, axis=1)
    ranked["__primary_l"] = ranked["__primary"].str.lower()

    selected_idx: List[int] = []
    used = set()

    def _pick_first(mask: pd.Series) -> None:
        for idx in ranked.index[mask]:
            if idx in used:
                continue
            used.add(idx)
            selected_idx.append(int(idx))
            return

    if requested_intents:
        for intent in requested_intents:
            _pick_first(ranked["__intent"] == intent)
    else:
        for tok in requested_tokens:
            _pick_first(ranked["__primary_l"].str.contains(tok, na=False))

    # Fill remaining positions by original ranking.
    for idx in ranked.index:
        if len(selected_idx) >= int(k):
            break
        if idx in used:
            continue
        used.add(idx)
        selected_idx.append(int(idx))

    out = ranked.loc[selected_idx].head(k).drop(columns=["__primary", "__intent", "__primary_l"], errors="ignore")
    return out.reset_index(drop=True)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import atan2, cos, radians, sin, sqrt

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
    return r * c


def _location_radius_profile(city_qid: Optional[str]) -> Tuple[float, float, float]:
    """
    Radius policy for location route:
    - start_km: strict local radius target
    - max_km: hard cap for gradual expansion
    - step_km: expansion step when not enough candidates
    """
    qid = str(city_qid or "").strip().upper()
    if qid in {"Q35765", "Q406"}:  # Osaka / Istanbul
        return 1.8, 4.0, 0.40
    if qid == "Q864965":  # Petaling Jaya (sparser spread)
        return 2.8, 6.0, 0.50
    return 2.0, 4.5, 0.40


def _coalesce_latlon_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize coordinate columns to canonical 'lat'/'lon'.
    Some upstream merges may produce lat_x/lat_y, lon_x/lon_y.
    """
    if df is None or df.empty:
        return df
    work = df.copy()
    if "lat" not in work.columns:
        for cand in ("lat_x", "lat_y", "latitude"):
            if cand in work.columns:
                work["lat"] = work[cand]
                break
    if "lon" not in work.columns:
        for cand in ("lon_x", "lon_y", "lng", "longitude"):
            if cand in work.columns:
                work["lon"] = work[cand]
                break
    return work


def _trim_location_nearest(
    df: pd.DataFrame,
    lat: Optional[float],
    lon: Optional[float],
    k: int,
    *,
    city_qid: Optional[str] = None,
) -> pd.DataFrame:
    """
    Make location route clearly proximity-dominant:
    - compute anchor distance for all candidates
    - sort primarily by distance (nearest first), then by score as tie-breaker
    """
    if df is None or df.empty:
        return df
    df = _coalesce_latlon_columns(df)
    if lat is None or lon is None:
        return df.head(k).reset_index(drop=True)
    if "lat" not in df.columns or "lon" not in df.columns:
        return df.head(k).reset_index(drop=True)

    work = df.copy()
    work = work.dropna(subset=["fsq_id", "lat", "lon"], how="any").reset_index(drop=True)
    if work.empty:
        return work

    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work["score"] = pd.to_numeric(work.get("score", 0.0), errors="coerce").fillna(0.0)
    work = work.dropna(subset=["lat", "lon"], how="any").reset_index(drop=True)
    if work.empty:
        return work

    # Anchor distance for all candidates.
    work["anchor_distance_km"] = work.apply(
        lambda r: _haversine_km(float(lat), float(lon), float(r["lat"]), float(r["lon"])),
        axis=1,
    )
    work["distance_km"] = work["anchor_distance_km"]

    # Radius gating: start strict, then expand progressively only if needed.
    # If we still do not have enough candidates after max radius, we keep the
    # available local subset instead of pulling distant points.
    start_km, max_km, step_km = _location_radius_profile(city_qid)
    radius = float(start_km)
    gated = work[work["anchor_distance_km"] <= radius].copy()
    while len(gated) < int(k) and radius < float(max_km):
        radius = min(float(max_km), radius + float(step_km))
        gated = work[work["anchor_distance_km"] <= radius].copy()
    if gated.empty:
        return df.head(k).reset_index(drop=True)
    work = gated

    # Normalize model score for combination with geographic terms.
    smin = float(work["score"].min())
    smax = float(work["score"].max())
    if smax - smin <= 1e-12:
        work["model_norm"] = 0.0
    else:
        work["model_norm"] = (work["score"] - smin) / (smax - smin)

    # Explicit distance penalty from anchor (strong locality control).
    # Larger lambda => faster score decay with distance.
    # 3.2 intentionally makes location route hyper-local.
    work["dist_penalty"] = work["anchor_distance_km"].apply(lambda d: float(pow(2.718281828, -3.2 * float(d))))
    # Keep a user-facing score already locality-penalized.
    work["score"] = work["model_norm"] * work["dist_penalty"]

    def _anchor_pref(d: float) -> float:
        # Strong near-anchor preference for strict location mode.
        return 1.0 / (1.0 + (d / 1.4))

    def _leg_pref(d: float) -> float:
        # Prefer usable route legs (not too tiny, not huge jumps).
        target = 1.4
        sigma = 0.95
        base = float(pow(2.718281828, -((d - target) ** 2) / (2.0 * sigma * sigma)))
        if d < 0.25:
            base *= 0.15
        elif d < 0.45:
            base *= 0.55
        if d > 5.5:
            base *= 0.5
        return base

    # Pick first point: strongly prioritize anchor proximity.
    work["first_score"] = (0.05 * work["score"]) + (0.95 * work["anchor_distance_km"].apply(_anchor_pref))
    selected_idx: List[int] = []
    used = set()
    first_idx = int(work["first_score"].idxmax())
    selected_idx.append(first_idx)
    used.add(str(work.loc[first_idx, "fsq_id"]))

    # Sequentially grow route with model + anchor + leg coherence + light diversity.
    while len(selected_idx) < int(k):
        prev = work.loc[selected_idx[-1]]
        prev_cat = str(prev.get("primary_category") or "").strip().lower()
        best_idx = None
        best_score = -1e9

        for idx, row in work.iterrows():
            fid = str(row.get("fsq_id") or "")
            if not fid or fid in used:
                continue
            leg_km = _haversine_km(float(prev["lat"]), float(prev["lon"]), float(row["lat"]), float(row["lon"]))
            m = float(row["score"])  # model already decayed by anchor distance
            a = _anchor_pref(float(row["anchor_distance_km"]))
            l = _leg_pref(float(leg_km))
            row_cat = str(row.get("primary_category") or "").strip().lower()
            cat_bonus = 0.05 if row_cat and row_cat != prev_cat else 0.0
            # Hyper-local route: anchor dominates; model acts as tie-breaker.
            total = (0.05 * m) + (0.85 * a) + (0.10 * l) + cat_bonus
            # Additional soft penalty near the current radius boundary.
            if float(row["anchor_distance_km"]) > (0.85 * radius):
                total *= 0.70
            # Prevent very long hops in location-only mode.
            if leg_km > 1.2:
                total *= 0.20
            if total > best_score:
                best_score = total
                best_idx = int(idx)

        if best_idx is None:
            break
        selected_idx.append(best_idx)
        used.add(str(work.loc[best_idx, "fsq_id"]))

    out = work.loc[selected_idx].copy()
    if len(out) < int(k):
        # Keep strict locality: fill only from remaining points inside radius.
        remain = work[~work["fsq_id"].astype(str).isin(set(out["fsq_id"].astype(str)))]
        remain = remain.sort_values(["anchor_distance_km", "score"], ascending=[True, False])
        out = pd.concat([out, remain], ignore_index=True)
    out = out.sort_values(["anchor_distance_km", "score"], ascending=[True, False]).head(int(k)).reset_index(drop=True)
    return out


def _location_leg_cap_km(city_qid: Optional[str], prioritize_proximity: bool) -> float:
    qid = str(city_qid or "").strip().upper()
    if qid in {"Q35765", "Q406"}:
        return 2.2 if prioritize_proximity else 3.0
    if qid == "Q864965":
        return 3.2 if prioritize_proximity else 4.2
    return 2.6 if prioritize_proximity else 3.6


def _location_anchor_target_km(city_qid: Optional[str], prioritize_proximity: bool) -> float:
    """
    Preferred distance-to-anchor for location routes.
    Helps avoid degenerate ultra-tiny clusters right on top of the anchor.
    """
    qid = str(city_qid or "").strip().upper()
    if qid in {"Q35765", "Q406"}:
        return 1.0 if prioritize_proximity else 1.6
    if qid == "Q864965":
        return 1.6 if prioritize_proximity else 2.4
    return 1.2 if prioritize_proximity else 1.9


def _build_location_route(
    *,
    dsn: Optional[str],
    city: Optional[str],
    city_qid: Optional[str],
    user_id: Optional[int],
    current_poi: Optional[str],
    lat: float,
    lon: float,
    k: int,
    visits_limit: Optional[int],
    use_embeddings: bool,
    embeddings_path: Optional[str],
    use_als: bool,
    als_path: Optional[str],
    prioritize_proximity: bool,
) -> pd.DataFrame:
    # 1) Geo-first candidate pool from POIs in city.
    conn = get_conn(dsn)
    try:
        pois = load_pois(conn, city=city, city_qid=city_qid)
    finally:
        conn.close()
    if pois is None or pois.empty:
        return pd.DataFrame()

    work = _coalesce_latlon_columns(pois)
    if "lat" not in work.columns or "lon" not in work.columns:
        return pd.DataFrame()
    work = work.dropna(subset=["fsq_id", "lat", "lon", "name", "primary_category"], how="any").copy()
    if work.empty:
        return pd.DataFrame()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.dropna(subset=["lat", "lon"], how="any")
    if work.empty:
        return pd.DataFrame()

    work["anchor_distance_km"] = work.apply(
        lambda r: _haversine_km(float(lat), float(lon), float(r["lat"]), float(r["lon"])),
        axis=1,
    )
    work["distance_km"] = work["anchor_distance_km"]

    # 2) Hard local radius (expands only a bit; never goes far).
    start_km, max_km, step_km = _location_radius_profile(city_qid)
    if prioritize_proximity:
        start_km *= 0.85
        max_km *= 0.85
    radius = float(start_km)
    gated = work[work["anchor_distance_km"] <= radius].copy()
    target_pool = int(max(k * 6, 80))
    while len(gated) < target_pool and radius < float(max_km):
        radius = min(float(max_km), radius + float(step_km))
        gated = work[work["anchor_distance_km"] <= radius].copy()
    if gated.empty:
        return pd.DataFrame()
    work = gated.copy()

    # 3) Secondary model signal (independent from full route composition).
    # Keep location independent from user-history personalization.
    model_k = int(max(k * 20, 500))
    model_df = recommend(
        dsn=dsn,
        city=city,
        city_qid=city_qid,
        user_id=None,
        current_poi=current_poi,
        k=model_k,
        visits_limit=visits_limit,
        mode="hybrid",
        use_embeddings=use_embeddings,
        embeddings_path=embeddings_path,
        use_als=use_als,
        als_path=als_path,
        lat=None,
        lon=None,
        max_price_tier=None,
        free_only=False,
        prefs=None,
        distance_weight=0.0,
        diversify=True,
    )
    model_map: Dict[str, float] = {}
    if model_df is not None and not model_df.empty and "fsq_id" in model_df.columns:
        ms = model_df[["fsq_id", "score"]].copy()
        ms["score"] = pd.to_numeric(ms["score"], errors="coerce").fillna(0.0)
        smin, smax = float(ms["score"].min()), float(ms["score"].max())
        if smax - smin > 1e-12:
            ms["score_n"] = (ms["score"] - smin) / (smax - smin)
        else:
            ms["score_n"] = 0.0
        model_map = {str(r["fsq_id"]): float(r["score_n"]) for _, r in ms.iterrows()}
    work["model_norm"] = work["fsq_id"].astype(str).map(model_map).fillna(0.0).astype(float)

    # Quality prior from POI metadata.
    work["rating"] = pd.to_numeric(work.get("rating"), errors="coerce").fillna(0.0)
    work["total_ratings"] = pd.to_numeric(work.get("total_ratings"), errors="coerce").fillna(0.0)
    rmax = float(work["rating"].max()) if len(work) else 0.0
    tmax = float(work["total_ratings"].max()) if len(work) else 0.0
    work["r_norm"] = (work["rating"] / rmax) if rmax > 0 else 0.0
    work["t_norm"] = (work["total_ratings"] / tmax) if tmax > 0 else 0.0
    work["quality_norm"] = 0.7 * work["r_norm"] + 0.3 * work["t_norm"]

    # 4) Local candidate score (geo-dominant).
    target_km = _location_anchor_target_km(city_qid, prioritize_proximity=prioritize_proximity)
    sigma = 0.85 if prioritize_proximity else 1.20
    # Ring-like preference: still local, but not all points collapsed at ~0 km.
    work["geo_pref"] = work["anchor_distance_km"].apply(
        lambda d: float(pow(2.718281828, -((float(d) - target_km) ** 2) / (2.0 * sigma * sigma)))
    )
    work["local_score"] = (
        (0.40 * work["geo_pref"]) + (0.45 * work["model_norm"]) + (0.15 * work["quality_norm"])
    )

    # 5) Greedy route build with hard local constraints.
    leg_cap = _location_leg_cap_km(city_qid, prioritize_proximity=prioritize_proximity)
    beta = 1.05 if prioritize_proximity else 0.80
    selected: List[int] = []
    used_ids = set()

    first_idx = int(work["local_score"].idxmax())
    selected.append(first_idx)
    used_ids.add(str(work.loc[first_idx, "fsq_id"]))

    while len(selected) < int(k):
        prev = work.loc[selected[-1]]
        prev_cat = str(prev.get("primary_category") or "").strip().lower()
        best_idx = None
        best_score = -1e18
        for idx, row in work.iterrows():
            fid = str(row.get("fsq_id") or "")
            if not fid or fid in used_ids:
                continue
            d_anchor = float(row["anchor_distance_km"])
            if d_anchor > float(radius):
                continue
            leg_km = _haversine_km(float(prev["lat"]), float(prev["lon"]), float(row["lat"]), float(row["lon"]))
            if leg_km > float(leg_cap):
                continue
            leg_pref = float(pow(2.718281828, -beta * leg_km))
            cat_bonus = 0.05 if str(row.get("primary_category") or "").strip().lower() != prev_cat else 0.0
            total = (0.35 * row["geo_pref"]) + (0.20 * leg_pref) + (0.45 * row["model_norm"]) + cat_bonus
            if total > best_score:
                best_score = float(total)
                best_idx = int(idx)
        if best_idx is None:
            break
        selected.append(best_idx)
        used_ids.add(str(work.loc[best_idx, "fsq_id"]))

    out = work.loc[selected].copy() if selected else pd.DataFrame()
    if out.empty:
        return out
    if len(out) < int(k):
        remain = work[~work["fsq_id"].astype(str).isin(set(out["fsq_id"].astype(str)))].copy()
        remain = remain.sort_values(["local_score", "anchor_distance_km"], ascending=[False, True])
        out = pd.concat([out, remain.head(int(k) - len(out))], ignore_index=True)
    out["score"] = out["local_score"]
    out = out.sort_values(["score", "anchor_distance_km"], ascending=[False, True]).head(int(k)).reset_index(drop=True)
    return out


def _norm_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 1e-12:
        return pd.Series([0.0] * len(s), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def _blend_routes(
    parts: List[Tuple[str, pd.DataFrame, float]],
    *,
    k: int,
) -> pd.DataFrame:
    """
    Weighted blend across route variants using normalized per-variant scores.
    Deduplicates by fsq_id and preserves representative row data.
    """
    agg: Dict[str, Dict[str, object]] = {}
    for _, df, w in parts:
        if df is None or df.empty or w <= 0:
            continue
        work = df.copy()
        if "fsq_id" not in work.columns:
            continue
        if "score" not in work.columns:
            work["score"] = 0.0
        work["__s"] = _norm_score(work["score"])
        for _, row in work.iterrows():
            pid = str(row.get("fsq_id") or "").strip()
            if not pid:
                continue
            contrib = float(w) * float(row.get("__s", 0.0))
            entry = agg.get(pid)
            if entry is None:
                rowd = row.to_dict()
                rowd.pop("__s", None)
                agg[pid] = {"row": rowd, "blend_score": contrib}
            else:
                entry["blend_score"] = float(entry["blend_score"]) + contrib
                if float(row.get("__s", 0.0)) > float(entry.get("best_local_s", -1.0)):
                    rowd = row.to_dict()
                    rowd.pop("__s", None)
                    entry["row"] = rowd
                entry["best_local_s"] = max(float(entry.get("best_local_s", -1.0)), float(row.get("__s", 0.0)))

    if not agg:
        return pd.DataFrame()

    rows = []
    for val in agg.values():
        row = dict(val["row"])
        row["score"] = float(val["blend_score"])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out.head(int(k)).reset_index(drop=True)


def build_multi_routes(
    *,
    dsn: Optional[str] = None,
    city: Optional[str] = None,
    city_qid: Optional[str] = None,
    user_id: Optional[int] = None,
    current_poi: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    prefs: Optional[Prefs] = None,
    max_price_tier: Optional[int] = None,
    free_only: bool = False,
    k: int = 10,
    visits_limit: Optional[int] = 50000,
    use_embeddings: bool = False,
    embeddings_path: Optional[str] = None,
    use_als: bool = False,
    als_path: Optional[str] = None,
    prioritize_proximity: bool = False,
) -> MultiRouteResult:
    # "location" signal means explicit geographic start from request lat/lon.
    # current_poi can still be used as contextual sequence signal, but does not
    # activate the location-only route by itself.
    location_present = lat is not None and lon is not None
    inputs_present = _has_inputs(prefs=prefs, free_only=free_only, max_price_tier=max_price_tier)

    user_exists = _user_exists(dsn=dsn, user_id=user_id, city_qid=city_qid)
    history_present = user_exists

    request_signals = {
        "history": history_present,
        "inputs": inputs_present,
        "location": location_present,
    }

    routes: Dict[str, pd.DataFrame] = {}
    omitted: Dict[str, str] = {}
    warnings: List[str] = []

    # Guard clause for new users with no inputs.
    if not history_present and not inputs_present and not location_present:
        raise ValueError("missing_input: user has no history and no request inputs (prefs/location).")

    # 1) history route: only historical signals, no location/prefs.
    if history_present:
        df = recommend(
            dsn=dsn,
            city=city,
            city_qid=city_qid,
            user_id=user_id,
            current_poi=None,
            k=k,
            visits_limit=visits_limit,
            mode="hybrid",
            use_embeddings=use_embeddings,
            embeddings_path=embeddings_path,
            use_als=use_als,
            als_path=als_path,
            lat=None,
            lon=None,
            max_price_tier=None,
            free_only=False,
            prefs=None,
            distance_weight=0.0,
            diversify=True,
        )
        if df.empty:
            omitted["history"] = "empty_result"
        else:
            routes["history"] = df
    else:
        omitted["history"] = "user_without_history"

    # 2) inputs route: inputs dominant, isolated from history.
    if inputs_present:
        k_candidates = int(max(k * 6, min(300, k + 60)))
        df_soft = recommend(
            dsn=dsn,
            city=city,
            city_qid=city_qid,
            user_id=None,
            current_poi=None,
            k=k_candidates,
            visits_limit=visits_limit,
            mode="content",
            use_embeddings=False,
            embeddings_path=embeddings_path,
            use_als=False,
            als_path=als_path,
            lat=None,
            lon=None,
            max_price_tier=max_price_tier,
            free_only=free_only,
            prefs=prefs,
            distance_weight=0.0,
            diversify=True,
        )

        # Second pass (strict) to increase chance of including explicit user categories/intents.
        df_strict = pd.DataFrame()
        if prefs is not None and prefs.categories:
            strict_prefs = Prefs(
                categories=list(prefs.categories),
                free_only=prefs.free_only,
                max_price_tier=prefs.max_price_tier,
                category_mode="strict",
            )
            strict_k = int(max(k * 12, 400))
            df_strict = recommend(
                dsn=dsn,
                city=city,
                city_qid=city_qid,
                user_id=None,
                current_poi=None,
                k=strict_k,
                visits_limit=visits_limit,
                mode="content",
                use_embeddings=False,
                embeddings_path=embeddings_path,
                use_als=False,
                als_path=als_path,
                lat=None,
                lon=None,
                max_price_tier=max_price_tier,
                free_only=free_only,
                prefs=strict_prefs,
                distance_weight=0.0,
                diversify=True,
            )

        df = pd.concat([df_strict, df_soft], ignore_index=True)
        if not df.empty and "fsq_id" in df.columns:
            df = df.drop_duplicates(subset=["fsq_id"], keep="first").reset_index(drop=True)

        if df.empty:
            omitted["inputs"] = "empty_result"
        else:
            routes["inputs"] = _trim_inputs_with_category_coverage(df=df, prefs=prefs, k=k)
    else:
        omitted["inputs"] = "missing_inputs"

    # 3) location route: location dominant, independent from history.
    if location_present:
        df = _build_location_route(
            dsn=dsn,
            city=city,
            city_qid=city_qid,
            user_id=user_id if history_present else None,
            current_poi=current_poi,
            lat=float(lat),
            lon=float(lon),
            k=int(k),
            visits_limit=visits_limit,
            use_embeddings=False,
            embeddings_path=embeddings_path,
            use_als=False,
            als_path=als_path,
            prioritize_proximity=prioritize_proximity,
        )
        if df.empty:
            omitted["location"] = "empty_result"
        else:
            routes["location"] = df
    else:
        omitted["location"] = "missing_location"

    # 4) full route: use all available request signals (do not require all 3).
    full_mode = "hybrid"
    if inputs_present and not history_present and not location_present:
        # If only inputs exist, keep full aligned with input-dominant behavior.
        full_mode = "content"

    full_distance_weight = 0.0
    if location_present:
        full_distance_weight = 0.45 if prioritize_proximity else 0.30

    full_pool_k = int(max(int(k), min(400, max(int(k) * 8, int(k) + 80))))
    df = recommend(
        dsn=dsn,
        city=city,
        city_qid=city_qid,
        user_id=user_id if history_present else None,
        current_poi=current_poi,
        k=full_pool_k,
        visits_limit=visits_limit,
        mode=full_mode,
        use_embeddings=use_embeddings,
        embeddings_path=embeddings_path,
        use_als=use_als if history_present else False,
        als_path=als_path,
        lat=lat if location_present else None,
        lon=lon if location_present else None,
        max_price_tier=max_price_tier if inputs_present else None,
        free_only=free_only if inputs_present else False,
        prefs=prefs if inputs_present else None,
        distance_weight=full_distance_weight,
        diversify=True,
    )

    # Prevent "full" collapsing into a copy of "location" when several signals exist.
    blend_parts: List[Tuple[str, pd.DataFrame, float]] = []
    if history_present and "history" in routes:
        hist_w = 0.45 if (inputs_present and location_present) else 0.65
        if prioritize_proximity and location_present:
            hist_w *= 0.85
        blend_parts.append(("history", routes["history"], hist_w))
    if inputs_present and "inputs" in routes:
        inp_w = 0.35 if (history_present and location_present) else 0.55
        if prioritize_proximity and location_present:
            inp_w *= 0.9
        blend_parts.append(("inputs", routes["inputs"], inp_w))
    if location_present and "location" in routes:
        loc_w = 0.10 if (history_present and inputs_present) else 0.20
        if prioritize_proximity:
            loc_w *= 1.20
        blend_parts.append(("location", routes["location"], loc_w))

    if len(blend_parts) >= 2:
        blended = _blend_routes(blend_parts, k=full_pool_k)
        if not blended.empty:
            df = blended

    if df.empty:
        omitted["full"] = "empty_result"
    else:
        df_full = df.head(int(k)).reset_index(drop=True)
        df_full, surprise_used = _inject_soft_surprise(
            route_df=df_full,
            pool_df=df,
            k=int(k),
            prefs=prefs,
            lat=lat,
            lon=lon,
            city_qid=city_qid,
            user_id=user_id if history_present else None,
        )
        if surprise_used:
            warnings.append("soft_surprise_injected:full")
        elif "is_surprise" not in df_full.columns:
            df_full["is_surprise"] = False
        routes["full"] = df_full

    if not routes:
        warnings.append("no_routes_generated")

    return MultiRouteResult(
        request_signals=request_signals,
        user_exists=user_exists,
        routes=routes,
        omitted=omitted,
        warnings=warnings,
    )


__all__ = ["MultiRouteResult", "build_multi_routes"]
