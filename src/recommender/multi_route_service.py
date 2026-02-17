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
from typing import Dict, List, Optional

import pandas as pd

from .prefs import Prefs
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
        df = recommend(
            dsn=dsn,
            city=city,
            city_qid=city_qid,
            user_id=None,
            current_poi=None,
            k=k,
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
        if df.empty:
            omitted["inputs"] = "empty_result"
        else:
            routes["inputs"] = df
    else:
        omitted["inputs"] = "missing_inputs"

    # 3) location route: location dominant, independent from history.
    if location_present:
        df = recommend(
            dsn=dsn,
            city=city,
            city_qid=city_qid,
            user_id=None,
            current_poi=current_poi,
            k=k,
            visits_limit=visits_limit,
            mode="hybrid",
            use_embeddings=use_embeddings,
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
            routes["location"] = df
    else:
        omitted["location"] = "missing_location"

    # 4) full route: use all available request signals (do not require all 3).
    full_mode = "hybrid"
    if inputs_present and not history_present and not location_present:
        # If only inputs exist, keep full aligned with input-dominant behavior.
        full_mode = "content"

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
        distance_weight=0.35 if location_present else 0.0,
        diversify=True,
    )
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
