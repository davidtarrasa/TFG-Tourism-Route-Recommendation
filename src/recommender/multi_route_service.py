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
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .category_intents import INCONCLUSIVE, classify_category_intent, infer_user_intents
from .prefs import Prefs, normalize_categories
from .scorer import recommend
from .utils_db import get_conn


@dataclass
class MultiRouteResult:
    request_signals: Dict[str, bool]
    user_exists: bool
    routes: Dict[str, pd.DataFrame]
    omitted: Dict[str, str]
    warnings: List[str]


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


def _trim_location_nearest(df: pd.DataFrame, lat: Optional[float], lon: Optional[float], k: int) -> pd.DataFrame:
    """
    Make location route clearly proximity-dominant:
    - compute anchor distance for all candidates
    - sort primarily by distance (nearest first), then by score as tie-breaker
    """
    if df is None or df.empty:
        return df
    if lat is None or lon is None:
        return df.head(k).reset_index(drop=True)
    if "lat" not in df.columns or "lon" not in df.columns:
        return df.head(k).reset_index(drop=True)

    work = df.copy()

    def _dist(row: pd.Series) -> float:
        try:
            la = float(row.get("lat"))
            lo = float(row.get("lon"))
        except Exception:
            return float("inf")
        return _haversine_km(float(lat), float(lon), la, lo)

    work["anchor_distance_km"] = work.apply(_dist, axis=1)
    if "distance_km" not in work.columns:
        work["distance_km"] = work["anchor_distance_km"]

    # Location route policy:
    # - prefer very close POIs first
    # - only expand radius if we cannot fill k
    # This keeps location route clearly local around the start point.
    ring_km = [1.2, 2.0, 3.0, 5.0, 8.0]
    selected_parts: List[pd.DataFrame] = []
    selected_ids = set()

    for r in ring_km:
        part = work[work["anchor_distance_km"] <= float(r)].copy()
        if part.empty:
            continue
        part = part.sort_values(["anchor_distance_km", "score"], ascending=[True, False])
        if selected_ids:
            part = part[~part["fsq_id"].astype(str).isin(selected_ids)]
        if part.empty:
            continue
        selected_parts.append(part)
        selected_ids.update(part["fsq_id"].astype(str).tolist())
        if sum(len(x) for x in selected_parts) >= int(k):
            break

    if selected_parts:
        out = pd.concat(selected_parts, ignore_index=True)
    else:
        out = work.copy()

    # If even with the widest ring we still have <k, fill by nearest overall.
    if len(out) < int(k):
        extra = work.sort_values(["anchor_distance_km", "score"], ascending=[True, False]).copy()
        extra = extra[~extra["fsq_id"].astype(str).isin(set(out["fsq_id"].astype(str).tolist()))]
        if not extra.empty:
            out = pd.concat([out, extra], ignore_index=True)

    out = out.sort_values(["anchor_distance_km", "score"], ascending=[True, False])
    return out.head(int(k)).reset_index(drop=True)


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
        k_candidates = int(max(k * 6, min(150, k + 30)))
        df = recommend(
            dsn=dsn,
            city=city,
            city_qid=city_qid,
            user_id=None,
            current_poi=None,
            k=k_candidates,
            visits_limit=visits_limit,
            mode="hybrid",
            use_embeddings=False,
            embeddings_path=embeddings_path,
            use_als=False,
            als_path=als_path,
            lat=lat,
            lon=lon,
            max_price_tier=None,
            free_only=False,
            prefs=None,
            distance_weight=0.9,
            diversify=True,
        )
        if df.empty:
            omitted["location"] = "empty_result"
        else:
            routes["location"] = _trim_location_nearest(df=df, lat=lat, lon=lon, k=k)
    else:
        omitted["location"] = "missing_location"

    # 4) full route: use all available request signals (do not require all 3).
    full_mode = "hybrid"
    if inputs_present and not history_present and not location_present:
        # If only inputs exist, keep full aligned with input-dominant behavior.
        full_mode = "content"

    full_distance_weight = 0.0
    if location_present:
        full_distance_weight = 0.6 if prioritize_proximity else 0.35

    df = recommend(
        dsn=dsn,
        city=city,
        city_qid=city_qid,
        user_id=user_id if history_present else None,
        current_poi=current_poi,
        k=k,
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
        loc_w = 0.20 if (history_present and inputs_present) else 0.35
        if prioritize_proximity:
            loc_w *= 1.75
        blend_parts.append(("location", routes["location"], loc_w))

    if len(blend_parts) >= 2:
        blended = _blend_routes(blend_parts, k=k)
        if not blended.empty:
            df = blended

    if df.empty:
        omitted["full"] = "empty_result"
    else:
        routes["full"] = df

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
