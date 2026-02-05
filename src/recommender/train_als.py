"""
Entrena un modelo ALS implícito y lo guarda en cache (joblib).

Uso:
  python -m src.recommender.train_als --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/als_osaka.joblib
"""

import argparse
import os
import joblib

from .utils_db import get_conn, load_visits
from .models.als import build_interactions, train_als


def main():
    parser = argparse.ArgumentParser(description="Entrena ALS implícito sobre visits")
    parser.add_argument("--city", help="Filtro opcional pois.city (no aplica a visits)")
    parser.add_argument("--city-qid", dest="city_qid", help="Filtro visits.venue_city")
    parser.add_argument("--visits-limit", type=int, dest="visits_limit", help="Limitar visits para acelerar")
    parser.add_argument("--out", default="src/recommender/cache/als_model.joblib", help="Ruta de salida")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=40.0)
    args = parser.parse_args()

    conn = get_conn()
    visits = load_visits(conn, city_qid=args.city_qid, limit=args.visits_limit)
    if visits.empty:
        raise SystemExit("No hay visits para entrenar.")

    interactions, user_to_idx, item_to_idx, idx_to_item = build_interactions(visits)
    model = train_als(interactions, factors=args.factors, regularization=args.reg, iterations=args.iters, alpha=args.alpha)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "idx_to_item": idx_to_item,
        },
        args.out,
    )
    print(f"Modelo ALS guardado en {args.out}")


if __name__ == "__main__":
    main()
