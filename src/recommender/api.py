"""
FastAPI wrapper for recommender services.

Endpoints:
- GET /health
- POST /recommend
- POST /multi-recommend
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_saved_routes_table(dsn: Optional[str] = None) -> None:
    conn = get_conn(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS saved_routes (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT,
                    city_qid TEXT,
                    route_type TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'frontend',
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_saved_routes_user_created ON saved_routes(user_id, created_at DESC);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_saved_routes_city_created ON saved_routes(city_qid, created_at DESC);"
            )
        conn.commit()
    finally:
        conn.close()


class SaveRouteRequest(BaseModel):
    dsn: Optional[str] = None
    user_id: Optional[int] = None
    city_qid: Optional[str] = None
    route_type: str
    source: str = "frontend"
    payload: Dict[str, Any]


class DeleteSavedRoutesRequest(BaseModel):
    dsn: Optional[str] = None
    user_id: Optional[int] = None
    city_qid: Optional[str] = None


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


@app.post("/saved-routes")
def save_route_endpoint(req: SaveRouteRequest) -> Dict[str, Any]:
    try:
        _ensure_saved_routes_table(req.dsn)
        conn = get_conn(req.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO saved_routes (user_id, city_qid, route_type, source, payload)
                    VALUES (%(user_id)s, %(city_qid)s, %(route_type)s, %(source)s, %(payload)s::jsonb)
                    RETURNING id, created_at
                    """,
                    {
                        "user_id": req.user_id,
                        "city_qid": req.city_qid,
                        "route_type": req.route_type,
                        "source": req.source,
                        "payload": json.dumps(req.payload, ensure_ascii=False),
                    },
                )
                row = cur.fetchone()
            conn.commit()
        finally:
            conn.close()
        return {"status": "ok", "id": int(row[0]), "created_at": str(row[1])}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"save_route_failed: {exc}") from exc


@app.get("/saved-routes")
def list_saved_routes(
    dsn: Optional[str] = None,
    user_id: Optional[int] = None,
    city_qid: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    if limit < 1:
        limit = 1
    if limit > 1000:
        limit = 1000
    try:
        _ensure_saved_routes_table(dsn)
        conn = get_conn(dsn)
        try:
            clauses = []
            params: Dict[str, Any] = {"limit": int(limit)}
            if user_id is not None:
                clauses.append("user_id = %(user_id)s")
                params["user_id"] = int(user_id)
            if city_qid is not None:
                clauses.append("city_qid = %(city_qid)s")
                params["city_qid"] = city_qid
            q = "SELECT id, user_id, city_qid, route_type, source, payload, created_at FROM saved_routes"
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " ORDER BY created_at DESC LIMIT %(limit)s"
            df = pd.read_sql(q, conn, params=params)
        finally:
            conn.close()
        return {"status": "ok", "count": int(len(df)), "items": _df_to_records(df)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"list_saved_routes_failed: {exc}") from exc


@app.delete("/saved-routes")
def delete_saved_routes(req: DeleteSavedRoutesRequest) -> Dict[str, Any]:
    try:
        _ensure_saved_routes_table(req.dsn)
        conn = get_conn(req.dsn)
        try:
            clauses = []
            params: Dict[str, Any] = {}
            if req.user_id is not None:
                clauses.append("user_id = %(user_id)s")
                params["user_id"] = int(req.user_id)
            if req.city_qid is not None:
                clauses.append("city_qid = %(city_qid)s")
                params["city_qid"] = req.city_qid

            q = "DELETE FROM saved_routes"
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " RETURNING id"
            with conn.cursor() as cur:
                cur.execute(q, params)
                deleted = cur.fetchall()
            conn.commit()
        finally:
            conn.close()
        return {"status": "ok", "deleted": len(deleted)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"delete_saved_routes_failed: {exc}") from exc
