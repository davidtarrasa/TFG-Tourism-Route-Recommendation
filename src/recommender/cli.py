"""
CLI de recomendación.
- Argumentos: --user-id, --city, --mode (hybrid/content/item/markov/embed), --k, --current-poi, --visits-limit, etc.
- Puede generar ruta ordenada (GeoJSON + HTML) con --build-route.
"""

import argparse
import sys

import pandas as pd

from .scorer import recommend
from .prefs import parse_prefs


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Recomendador en terminal (baseline híbrido/content/item/markov/embed)")
    parser.add_argument("--user-id", type=int, help="ID de usuario (si existe en visits)")
    parser.add_argument("--city", help="Filtro sobre pois.city (ej. Osaka)")
    parser.add_argument("--city-qid", dest="city_qid", help="Filtro sobre visits.venue_city (ej. Q406)")
    parser.add_argument("--current-poi", dest="current_poi", help="POI actual para modo Markov/distancia")
    parser.add_argument("--mode", choices=["hybrid", "content", "item", "markov", "embed", "als"], default="hybrid")
    parser.add_argument("--k", type=int, default=10, help="Número de resultados")
    parser.add_argument("--visits-limit", dest="visits_limit", type=int, default=50000, help="Límite de visits para features")
    parser.add_argument("--dsn", help="DSN de Postgres (si no, usa POSTGRES_DSN o por defecto)")
    parser.add_argument("--use-embeddings", action="store_true", help="Entrenar/cargar Word2Vec y usar vecinos")
    parser.add_argument("--embeddings-path", help="Ruta para cargar/guardar modelo Word2Vec", default="src/recommender/cache/word2vec.joblib")
    parser.add_argument("--use-als", action="store_true", help="Usar modelo ALS pre-entrenado (joblib)")
    parser.add_argument("--als-path", help="Ruta al modelo ALS (joblib)", default="src/recommender/cache/als_model.joblib")
    parser.add_argument("--build-route", action="store_true", help="Ordenar POIs y generar GeoJSON/HTML de la ruta")
    parser.add_argument("--route-output", default="data/reports/routes/route.html", help="Ruta de salida del mapa HTML")
    parser.add_argument("--geojson-output", default="data/reports/routes/route.geojson", help="Ruta de salida del GeoJSON")
    parser.add_argument("--lat", type=float, help="Latitud actual para re-ranking por distancia")
    parser.add_argument("--lon", type=float, help="Longitud actual para re-ranking por distancia")
    parser.add_argument("--max-price-tier", dest="max_price_tier", type=int, help="Filtrar por price_tier <= valor")
    parser.add_argument("--free-only", dest="free_only", action="store_true", help="Filtrar solo POIs gratuitos")
    parser.add_argument(
        "--prefs",
        help='Preferencias en una cadena separada por comas (ej: "museum,park,free,cheap")',
    )
    parser.add_argument("--distance-weight", dest="distance_weight", type=float, default=0.3, help="Peso penalización distancia [0-1]")
    parser.add_argument("--no-diversify", dest="diversify", action="store_false", help="Desactivar diversidad por categoría")
    args = parser.parse_args()

    prefs = parse_prefs(args.prefs)

    df = recommend(
        dsn=args.dsn,
        city=args.city,
        city_qid=args.city_qid,
        user_id=args.user_id,
        current_poi=args.current_poi,
        k=args.k,
        visits_limit=args.visits_limit,
        mode=args.mode,
        use_embeddings=args.use_embeddings,
        embeddings_path=args.embeddings_path,
        use_als=args.use_als,
        als_path=args.als_path,
        lat=args.lat,
        lon=args.lon,
        max_price_tier=args.max_price_tier,
        free_only=args.free_only,
        prefs=prefs,
        distance_weight=args.distance_weight,
        diversify=args.diversify,
    )

    if df.empty:
        print("No se encontraron recomendaciones (posible falta de historial o filtros muy restrictivos).")
        return

    # Mostrar resultados
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    if args.build_route and not df.empty:
        from .config import DEFAULT_CONFIG_PATH, load_config
        from .route_builder import build_route, to_geojson, to_folium_map
        from .route_planner import plan_route
        from .utils_db import get_conn, load_pois
        import os, json

        os.makedirs(os.path.dirname(args.route_output), exist_ok=True)
        os.makedirs(os.path.dirname(args.geojson_output), exist_ok=True)

        # Completar lat/lon si faltan
        if "lat" not in df.columns or "lon" not in df.columns:
            conn = get_conn(args.dsn)
            pois_full = load_pois(conn, city=args.city)
            df = df.merge(pois_full[["fsq_id", "lat", "lon"]], on="fsq_id", how="left")

        anchor = None
        if args.lat is not None and args.lon is not None:
            anchor = (args.lat, args.lon)

        # Build a better itinerary by selecting from a larger pool using route constraints.
        cfg = load_config(DEFAULT_CONFIG_PATH)
        route_cfg = cfg.get("route", {})
        route_pl_cfg = cfg.get("route_planner", {})

        pool_k = max(int(args.k) * 50, 500)
        try:
            df_pool = recommend(
                dsn=args.dsn,
                city=args.city,
                city_qid=args.city_qid,
                user_id=args.user_id,
                current_poi=args.current_poi,
                k=pool_k,
                visits_limit=args.visits_limit,
                mode=args.mode,
                use_embeddings=args.use_embeddings,
                embeddings_path=args.embeddings_path,
                use_als=args.use_als,
                als_path=args.als_path,
                lat=args.lat,
                lon=args.lon,
                max_price_tier=args.max_price_tier,
                free_only=args.free_only,
                prefs=prefs,
                distance_weight=args.distance_weight,
                diversify=False,
            )
        except Exception:
            df_pool = df

        planned = plan_route(
            df_pool,
            k=int(args.k),
            anchor=anchor,
            min_leg_km=float(route_cfg.get("min_leg_km", 0.3)),
            max_leg_km=float(route_cfg.get("max_leg_km", 5.0)),
            pair_min_km=float(route_pl_cfg.get("pair_min_km", 0.2)),
            max_per_category=int(route_pl_cfg.get("max_per_category", 2)),
            distance_weight=float(route_pl_cfg.get("distance_weight", 0.35)),
            diversity_bonus=float(route_pl_cfg.get("diversity_bonus", 0.05)),
        )

        if not planned.ordered_df.empty:
            ordered_df = planned.ordered_df
            total_km = planned.total_km
        else:
            # Fallback: just order the current top-K.
            rr = build_route(df, anchor_lat=anchor[0] if anchor else None, anchor_lon=anchor[1] if anchor else None)
            ordered_df = rr.ordered_df
            total_km = rr.total_km

        gj = to_geojson(ordered_df)
        with open(args.geojson_output, "w", encoding="utf-8") as f:
            json.dump(gj, f, ensure_ascii=False, indent=2)

        m = to_folium_map(ordered_df, anchor=anchor)
        m.save(args.route_output)
        print(f"Ruta ordenada guardada en {args.route_output} (GeoJSON en {args.geojson_output}), total {total_km:.2f} km")


if __name__ == "__main__":
    main()
