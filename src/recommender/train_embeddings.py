"""
Train a Word2Vec (skip-gram) model on visit sequences and save it to cache.

Keep CLI minimal: training hyperparameters live in `configs/recommender.toml`.

Usage:
  python -m src.recommender.train_embeddings --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/word2vec_osaka.joblib
"""

import argparse
import os

import joblib

from .config import DEFAULT_CONFIG_PATH, load_config
from .features.load_data import load_all
from .features.word2vec import sequences_from_visits
from .models.embeddings import train_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Word2Vec embeddings over visit sequences")
    parser.add_argument("--dsn", help="Postgres DSN (otherwise POSTGRES_DSN/DEFAULT)")
    parser.add_argument("--city", help="Optional filter over pois.city")
    parser.add_argument("--city-qid", dest="city_qid", help="Optional filter over visits.venue_city (QID)")
    parser.add_argument("--visits-limit", dest="visits_limit", type=int, help="Limit visits (speed)")
    parser.add_argument("--out", default="src/recommender/cache/word2vec.joblib", help="Output joblib path")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config TOML path")
    args = parser.parse_args()

    cfg = load_config(args.config, city_qid=args.city_qid)
    emb_cfg = cfg.get("embeddings", {})

    visits_df, _, _ = load_all(dsn=args.dsn, city=args.city, city_qid=args.city_qid, visits_limit=args.visits_limit)
    seqs = sequences_from_visits(visits_df)
    if not seqs:
        raise SystemExit("No sequences to train embeddings (empty visits or too strict filter).")

    model = train_embeddings(
        seqs,
        vector_size=int(emb_cfg.get("vector_size", 128)),
        window=int(emb_cfg.get("window", 15)),
        min_count=int(emb_cfg.get("min_count", 2)),
        workers=int(emb_cfg.get("workers", 4)),
        epochs=int(emb_cfg.get("epochs", 10)),
        negative=int(emb_cfg.get("negative", 5)),
        sample=float(emb_cfg.get("sample", 1e-3)),
        ns_exponent=float(emb_cfg.get("ns_exponent", 0.75)),
        hs=int(emb_cfg.get("hs", 0)),
        seed=int(emb_cfg.get("seed", 42)),
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Word2Vec model saved to {args.out}")


if __name__ == "__main__":
    main()
