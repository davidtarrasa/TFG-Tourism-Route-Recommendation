"""
Generate route HTML files for the TFG memoir by calling the /multi-recommend API.

This script calls the same endpoint that the web frontend uses, so the routes
are guaranteed to be identical to what the demo shows.

Usage (API must be running):
    python scripts/generate_demo_routes.py --city-qid Q35765 --user-id 2725 \
        --lat 34.6937 --lon 135.5023 --prefs "food,culture,shopping" --k 8 \
        --use-embeddings --use-als --out-dir data/reports/routes/memoria_demo

To start the API first:
    uvicorn src.recommender.api:app --reload
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommender.route_builder import to_standalone_html

CITY_NAMES = {
    "Q35765": "Osaka",
    "Q406": "Istanbul",
    "Q864965": "Petaling Jaya",
}


def call_api(api_url: str, payload: dict, timeout: int = 90) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        f"{api_url}/multi-recommend",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        print(f"\nError: no se pudo conectar con la API en {api_url}")
        print(f"Detalle: {exc}")
        print("\nAsegúrate de que la API está corriendo:")
        print("  uvicorn src.recommender.api:app --reload")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Genera HTMLs de rutas llamando a /multi-recommend (idéntico al frontend)"
    )
    p.add_argument("--api-url", default="http://localhost:8000", help="URL base de la API")
    p.add_argument("--city-qid", dest="city_qid", default="Q35765")
    p.add_argument("--user-id", dest="user_id", type=int, default=None)
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--prefs", default="", help='Ej: "food,culture,shopping"')
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--use-embeddings", dest="use_embeddings", action="store_true")
    p.add_argument("--use-als", dest="use_als", action="store_true")
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        default=os.path.join("data", "reports", "routes", "memoria_demo"),
    )
    p.add_argument("--visits-limit", dest="visits_limit", type=int, default=10000)
    args = p.parse_args()

    city_qid = args.city_qid
    city_name = CITY_NAMES.get(city_qid, city_qid)
    has_location = args.lat is not None and args.lon is not None

    payload = {
        "city_qid": city_qid,
        "user_id": args.user_id,
        "k": args.k,
        "lat": args.lat,
        "lon": args.lon,
        "prefs": args.prefs,
        "max_price_tier": 3,
        "free_only": False,
        "category_mode": "soft",
        "use_embeddings": args.use_embeddings,
        "embeddings_path": f"src/recommender/cache/word2vec_{city_qid.lower()}.joblib",
        "use_als": args.use_als,
        "als_path": f"src/recommender/cache/als_{city_qid.lower()}.joblib",
        "visits_limit": args.visits_limit,
        "build_route": True,
        "prioritize_proximity": has_location,
    }

    api_url = args.api_url.rstrip("/")
    print(f"Llamando a {api_url}/multi-recommend ...")
    print(f"  city_qid={city_qid}, user_id={args.user_id}, k={args.k}, lat={args.lat}, lon={args.lon}")

    data = call_api(api_url, payload)

    signals = data.get("signals", {})
    warnings = data.get("warnings", [])
    routes = data.get("routes", {})
    omitted = data.get("omitted", {})

    print(f"Señales activas: {signals}")
    if warnings:
        print(f"Warnings: {warnings}")
    if omitted:
        print(f"Omitidas: {omitted}")
    print(f"Variantes generadas: {list(routes.keys())}")

    os.makedirs(args.out_dir, exist_ok=True)

    for variant, route_data in routes.items():
        ordered_pois = route_data.get("route", {}).get("ordered_pois") or []
        if not ordered_pois:
            print(f"[{variant}] Sin ordered_pois — variante omitida o sin ruta")
            continue

        df = pd.DataFrame(ordered_pois)
        anchor = (
            (args.lat, args.lon)
            if has_location and variant in ("location", "full")
            else None
        )

        html_content = to_standalone_html(
            df,
            anchor=anchor,
            variant_name=variant,
            city_name=city_name,
        )
        out_path = os.path.join(args.out_dir, f"route_{variant}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        total_km = route_data.get("route", {}).get("total_km", "?")
        print(f"[{variant}] {len(ordered_pois)} paradas - {total_km:.2f} km -> {out_path}")

    print("\nListo.")


if __name__ == "__main__":
    main()
