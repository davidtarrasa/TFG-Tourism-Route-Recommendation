"""
FastAPI wrapper for recommender services.

Endpoints:
- GET /health
- POST /recommend
- POST /multi-recommend
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .multi_route_service import build_multi_routes
from .prefs import parse_prefs
from .route_builder import build_route, to_geojson
from .scorer import recommend
from .utils_db import get_conn, load_pois


class RecommendRequest(BaseModel):
    dsn: Optional[str] = None
    city: Optional[str] = None
    city_qid: Optional[str] = None
    user_id: Optional[int] = None
    current_poi: Optional[str] = None
    mode: Literal["hybrid", "content", "item", "markov", "embed", "als"] = "hybrid"
    k: int = Field(default=10, ge=1, le=200)
    visits_limit: Optional[int] = Field(default=50000, ge=1)
    use_embeddings: bool = False
    embeddings_path: Optional[str] = "src/recommender/cache/word2vec.joblib"
    use_als: bool = False
    als_path: Optional[str] = "src/recommender/cache/als_model.joblib"
    lat: Optional[float] = None
    lon: Optional[float] = None
    max_price_tier: Optional[int] = Field(default=None, ge=0, le=4)
    free_only: bool = False
    prefs: Optional[str] = None
    category_mode: Literal["soft", "strict"] = "soft"
    distance_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    diversify: bool = True
    build_route: bool = False


class MultiRecommendRequest(BaseModel):
    dsn: Optional[str] = None
    city: Optional[str] = None
    city_qid: Optional[str] = None
    user_id: Optional[int] = None
    current_poi: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    prefs: Optional[str] = None
    category_mode: Literal["soft", "strict"] = "soft"
    max_price_tier: Optional[int] = Field(default=None, ge=0, le=4)
    free_only: bool = False
    k: int = Field(default=10, ge=1, le=200)
    visits_limit: Optional[int] = Field(default=50000, ge=1)
    use_embeddings: bool = False
    embeddings_path: Optional[str] = "src/recommender/cache/word2vec.joblib"
    use_als: bool = False
    als_path: Optional[str] = "src/recommender/cache/als_model.joblib"
    build_route: bool = False


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        out.append({k: _to_jsonable(v) for k, v in row.items()})
    return out


def _ensure_latlon(
    df: pd.DataFrame,
    *,
    dsn: Optional[str],
    city: Optional[str],
    city_qid: Optional[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    if "lat" in df.columns and "lon" in df.columns:
        return df
    conn = get_conn(dsn)
    pois = load_pois(conn, city=city, city_qid=city_qid)
    return df.merge(pois[["fsq_id", "lat", "lon"]], on="fsq_id", how="left")


def _route_payload(
    df: pd.DataFrame,
    *,
    dsn: Optional[str],
    city: Optional[str],
    city_qid: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> Dict[str, Any]:
    with_coords = _ensure_latlon(df, dsn=dsn, city=city, city_qid=city_qid)
    rr = build_route(with_coords, anchor_lat=lat, anchor_lon=lon)
    return {
        "total_km": float(rr.total_km),
        "ordered_pois": _df_to_records(rr.ordered_df),
        "geojson": to_geojson(rr.ordered_df),
    }


app = FastAPI(title="Tourism Route Recommender API", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest) -> Dict[str, Any]:
    prefs = parse_prefs(req.prefs, category_mode=req.category_mode)
    try:
        df = recommend(
            dsn=req.dsn,
            city=req.city,
            city_qid=req.city_qid,
            user_id=req.user_id,
            current_poi=req.current_poi,
            k=req.k,
            visits_limit=req.visits_limit,
            mode=req.mode,
            use_embeddings=req.use_embeddings,
            embeddings_path=req.embeddings_path,
            use_als=req.use_als,
            als_path=req.als_path,
            lat=req.lat,
            lon=req.lon,
            max_price_tier=req.max_price_tier,
            free_only=req.free_only,
            prefs=prefs,
            distance_weight=req.distance_weight,
            diversify=req.diversify,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"recommend_failed: {exc}") from exc

    payload: Dict[str, Any] = {"mode": req.mode, "k": req.k, "results": _df_to_records(df)}
    if req.build_route and not df.empty:
        payload["route"] = _route_payload(
            df,
            dsn=req.dsn,
            city=req.city,
            city_qid=req.city_qid,
            lat=req.lat,
            lon=req.lon,
        )
    return payload


@app.post("/multi-recommend")
def multi_recommend_endpoint(req: MultiRecommendRequest) -> Dict[str, Any]:
    prefs = parse_prefs(req.prefs, category_mode=req.category_mode)
    try:
        res = build_multi_routes(
            dsn=req.dsn,
            city=req.city,
            city_qid=req.city_qid,
            user_id=req.user_id,
            current_poi=req.current_poi,
            lat=req.lat,
            lon=req.lon,
            prefs=prefs,
            max_price_tier=req.max_price_tier,
            free_only=req.free_only,
            k=req.k,
            visits_limit=req.visits_limit,
            use_embeddings=req.use_embeddings,
            embeddings_path=req.embeddings_path,
            use_als=req.use_als,
            als_path=req.als_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"multi_recommend_failed: {exc}") from exc

    routes_payload: Dict[str, Any] = {}
    for key, df in res.routes.items():
        route_data: Dict[str, Any] = {"results": _df_to_records(df)}
        if req.build_route and not df.empty:
            route_data["route"] = _route_payload(
                df,
                dsn=req.dsn,
                city=req.city,
                city_qid=req.city_qid,
                lat=req.lat if key in ("location", "full") else None,
                lon=req.lon if key in ("location", "full") else None,
            )
        routes_payload[key] = route_data

    return {
        "signals": res.request_signals,
        "user_exists": res.user_exists,
        "routes": routes_payload,
        "omitted": res.omitted,
        "warnings": res.warnings,
    }

