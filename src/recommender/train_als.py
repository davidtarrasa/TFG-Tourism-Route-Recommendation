"""
Train an implicit ALS model and save it to cache (joblib).

Keep CLI minimal: training hyperparameters live in `configs/recommender.toml`.

Usage:
  python -m src.recommender.train_als --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/als_osaka.joblib
"""

import argparse
import os

import joblib

from .config import DEFAULT_CONFIG_PATH, load_config
from .models.als import build_interactions, train_als
from .utils_db import get_conn, load_visits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train implicit ALS over visits")
    parser.add_argument("--dsn", help="Postgres DSN (otherwise POSTGRES_DSN/DEFAULT)")
    parser.add_argument("--city-qid", dest="city_qid", help="Filter visits.venue_city (QID)")
    parser.add_argument("--visits-limit", type=int, dest="visits_limit", help="Limit visits (speed)")
    parser.add_argument("--out", default="src/recommender/cache/als_model.joblib", help="Output joblib path")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config TOML path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    als_cfg = cfg.get("als", {})

    conn = get_conn(args.dsn)
    visits = load_visits(conn, city_qid=args.city_qid, limit=args.visits_limit)
    if visits.empty:
        raise SystemExit("No visits to train ALS (empty or too strict filter).")

    interactions, user_to_idx, item_to_idx, idx_to_item = build_interactions(visits)
    model = train_als(
        interactions,
        factors=int(als_cfg.get("factors", 128)),
        regularization=float(als_cfg.get("regularization", 0.01)),
        iterations=int(als_cfg.get("iterations", 30)),
        alpha=float(als_cfg.get("alpha", 40.0)),
    )

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
    print(f"ALS model saved to {args.out}")


if __name__ == "__main__":
    main()

