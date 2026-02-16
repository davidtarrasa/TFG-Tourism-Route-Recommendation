"""
CLI to test the multi-route product contract in one request.

It can output:
- JSON summary (signals, generated/omitted routes)
- Per-route tables in stdout
- Optional map files (HTML + GeoJSON) for each generated variant
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import pandas as pd

from .multi_route_service import build_multi_routes
from .prefs import parse_prefs
from .route_builder import build_route, to_folium_map, to_geojson
from .route_planner import plan_route
from .utils_db import get_conn, load_pois


def _ensure_latlon(df: pd.DataFrame, dsn: Optional[str], city: Optional[str], city_qid: Optional[str]) -> pd.DataFrame:
    if "lat" in df.columns and "lon" in df.columns:
        return df
    conn = get_conn(dsn)
    pois = load_pois(conn, city=city, city_qid=city_qid)
    return df.merge(pois[["fsq_id", "lat", "lon"]], on="fsq_id", how="left")


def _build_route_outputs(
    *,
    df: pd.DataFrame,
    dsn: Optional[str],
    city: Optional[str],
    city_qid: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    out_html: str,
    out_geojson: str,
    k: int,
) -> Tuple[str, str]:
    from .config import DEFAULT_CONFIG_PATH, load_config

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    os.makedirs(os.path.dirname(out_geojson), exist_ok=True)

    cfg = load_config(DEFAULT_CONFIG_PATH, city_qid=city_qid)
    route_cfg = cfg.get("route", {})
    route_pl_cfg = cfg.get("route_planner", {})

    df = _ensure_latlon(df, dsn=dsn, city=city, city_qid=city_qid)
    anchor = (lat, lon) if (lat is not None and lon is not None) else None

    planned = plan_route(
        df,
        k=int(k),
        anchor=anchor,
        min_leg_km=float(route_cfg.get("min_leg_km", 0.3)),
        max_leg_km=float(route_cfg.get("max_leg_km", 5.0)),
        pair_min_km=float(route_pl_cfg.get("pair_min_km", 0.2)),
        max_per_category=int(route_pl_cfg.get("max_per_category", 2)),
        distance_weight=float(route_pl_cfg.get("distance_weight", 0.35)),
        distance_weight_no_anchor=float(route_pl_cfg.get("distance_weight_no_anchor"))
        if route_pl_cfg.get("distance_weight_no_anchor") is not None
        else None,
        max_leg_km_no_anchor=float(route_pl_cfg.get("max_leg_km_no_anchor"))
        if route_pl_cfg.get("max_leg_km_no_anchor") is not None
        else None,
        diversity_bonus=float(route_pl_cfg.get("diversity_bonus", 0.05)),
    )

    if not planned.ordered_df.empty:
        rr = build_route(planned.ordered_df, anchor_lat=anchor[0] if anchor else None, anchor_lon=anchor[1] if anchor else None)
        ordered_df = rr.ordered_df
    else:
        rr = build_route(df.head(k), anchor_lat=anchor[0] if anchor else None, anchor_lon=anchor[1] if anchor else None)
        ordered_df = rr.ordered_df

    gj = to_geojson(ordered_df)
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, indent=2)

    m = to_folium_map(ordered_df, anchor=anchor, route_modes=("drive",))
    m.save(out_html)
    return out_html, out_geojson


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Test multi-route contract (history/inputs/location/full)")
    p.add_argument("--dsn", help="Postgres DSN (fallback env/default)")
    p.add_argument("--city", help="Filter by pois.city")
    p.add_argument("--city-qid", dest="city_qid", help="Filter by city QID")
    p.add_argument("--user-id", type=int, help="User identifier")
    p.add_argument("--current-poi", dest="current_poi", help="Current POI")
    p.add_argument("--lat", type=float, help="Current latitude")
    p.add_argument("--lon", type=float, help="Current longitude")
    p.add_argument("--prefs", help='Comma-separated prefs, e.g. "museum,park,free,cheap"')
    p.add_argument("--max-price-tier", dest="max_price_tier", type=int, help="price_tier <= N")
    p.add_argument("--free-only", dest="free_only", action="store_true", help="Only free POIs")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--visits-limit", dest="visits_limit", type=int, default=50000)
    p.add_argument("--use-embeddings", action="store_true")
    p.add_argument("--embeddings-path", default="src/recommender/cache/word2vec.joblib")
    p.add_argument("--use-als", action="store_true")
    p.add_argument("--als-path", default="src/recommender/cache/als_model.joblib")
    p.add_argument("--build-route", action="store_true", help="Generate HTML/GeoJSON for each generated route variant")
    p.add_argument("--out-dir", default=os.path.join("data", "reports", "routes", "multi_route"))
    p.add_argument("--out-json", default=os.path.join("data", "reports", "multi_route_result.json"))
    args = p.parse_args()

    prefs = parse_prefs(args.prefs)
    result = build_multi_routes(
        dsn=args.dsn,
        city=args.city,
        city_qid=args.city_qid,
        user_id=args.user_id,
        current_poi=args.current_poi,
        lat=args.lat,
        lon=args.lon,
        prefs=prefs,
        max_price_tier=args.max_price_tier,
        free_only=args.free_only,
        k=args.k,
        visits_limit=args.visits_limit,
        use_embeddings=args.use_embeddings,
        embeddings_path=args.embeddings_path,
        use_als=args.use_als,
        als_path=args.als_path,
    )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    payload: Dict[str, object] = {
        "signals": result.request_signals,
        "user_exists": result.user_exists,
        "generated_routes": list(result.routes.keys()),
        "omitted": result.omitted,
        "warnings": result.warnings,
        "maps": {},
    }

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("Signals:", result.request_signals)
        print("User exists:", result.user_exists)
        print("Generated:", list(result.routes.keys()))
        print("Omitted:", result.omitted)
        for name, df in result.routes.items():
            print(f"\n=== Route variant: {name} ===")
            print(df.head(args.k))

    if args.build_route:
        os.makedirs(args.out_dir, exist_ok=True)
        for name, df in result.routes.items():
            html_path = os.path.join(args.out_dir, f"route_{name}.html")
            geo_path = os.path.join(args.out_dir, f"route_{name}.geojson")
            out_html, out_geo = _build_route_outputs(
                df=df,
                dsn=args.dsn,
                city=args.city,
                city_qid=args.city_qid,
                lat=args.lat,
                lon=args.lon,
                out_html=html_path,
                out_geojson=geo_path,
                k=args.k,
            )
            payload["maps"][name] = {"html": out_html, "geojson": out_geo}
            print(f"[{name}] map: {out_html}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary: {args.out_json}")


if __name__ == "__main__":
    main()
