"""Tune embeddings *scoring* parameters (no re-training).

We assume a Word2Vec model already exists on disk. This script tunes:
- context_n (how many last POIs to average)
- topn_score (neighbors per query)
- hub_alpha (penalize popular items in neighbor list)

Search space defaults live in `configs/recommender.toml` under `[tune.embeddings_scoring]`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional

import joblib

from .config import DEFAULT_CONFIG_PATH, load_config
from .tune.common import (
    TrailCase,
    apply_hub_penalty,
    embedding_context,
    eval_cases,
    make_train_test_trails,
    normalize_scores,
)
from .models.embeddings import score_embeddings_context
from .utils_db import get_conn, load_pois, load_visits


def main() -> None:
    p = argparse.ArgumentParser(description="Tune embeddings scoring params (trail protocol)")
    p.add_argument("--city-qid", required=True)
    p.add_argument("--embeddings-path", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--out", default=os.path.join("data", "reports", "tune_embeddings_scoring.json"))

    # Optional overrides (otherwise use [tune.data])
    p.add_argument("--visits-limit", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--min-train", type=int, default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    tune = cfg.get("tune", {})
    tune_data = tune.get("data", {})
    tune_e = tune.get("embeddings_scoring", {})

    visits_limit = int(args.visits_limit or tune_data.get("visits_limit", 120000))
    k = int(args.k or tune_data.get("k", 20))
    test_size = int(args.test_size or tune_data.get("test_size", 1))
    min_train = int(args.min_train or tune_data.get("min_train", 2))
    max_cases = int(args.max_cases or tune_data.get("max_cases", 150))
    seed = int(args.seed or tune.get("seed", 42))

    context_n_values = [int(x) for x in tune_e.get("context_n_values", [1, 2, 3])]
    topn_score_values = [int(x) for x in tune_e.get("topn_score_values", [500, 1000, 2000])]
    hub_alpha_values = [float(x) for x in tune_e.get("hub_alpha_values", [0.0, 0.2, 0.5])]

    rng = random.Random(seed)

    conn = get_conn()
    visits = load_visits(conn, city_qid=args.city_qid, limit=visits_limit)
    pois = load_pois(conn, city_qid=args.city_qid)

    candidate_set = set([str(x) for x in pois["fsq_id"].dropna().astype(str).tolist()])
    visits = visits[visits["venue_id"].astype(str).isin(candidate_set)].copy()
    if visits.empty:
        raise SystemExit("No visits after filtering (check city_qid and data load).")

    train_df, test_df, cases = make_train_test_trails(visits, test_size=test_size, min_train=min_train)
    rng.shuffle(cases)
    cases = cases[:max_cases]
    if not cases:
        raise SystemExit("No evaluation cases. Try lowering min_train or increasing visits_limit.")

    item_counts = train_df["venue_id"].astype(str).value_counts().to_dict()
    model = joblib.load(args.embeddings_path)

    trials: List[Dict[str, object]] = []
    best = None
    best_key = (-1.0, -1.0)

    for context_n in context_n_values:
        for topn_score in topn_score_values:
            for hub_alpha in hub_alpha_values:

                def _scorer(c: TrailCase) -> Dict[str, float]:
                    seen = set([str(x) for x in c.user_items])
                    seen.add(str(c.current_poi))
                    if c.prev_poi:
                        seen.add(str(c.prev_poi))

                    ctx = embedding_context(c.user_items, c.current_poi, context_n=context_n)
                    eb = score_embeddings_context(model, ctx, topn=min(int(topn_score), 5000))
                    eb = {k: v for k, v in eb.items() if k in candidate_set and k not in seen}
                    eb = normalize_scores(apply_hub_penalty(eb, item_counts, hub_alpha))
                    return eb

                m = eval_cases(cases, k=k, scorer_fn=_scorer)
                row = {"context_n": context_n, "topn_score": topn_score, "hub_alpha": hub_alpha, "metrics": m}
                trials.append(row)
                key = (m["ndcg"], m["mrr"])
                if key > best_key:
                    best_key = key
                    best = row

    out = {
        "city_qid": args.city_qid,
        "embeddings_path": args.embeddings_path,
        "k": k,
        "visits_limit": visits_limit,
        "test_size": test_size,
        "min_train": min_train,
        "max_cases": max_cases,
        "search_space": {
            "context_n_values": context_n_values,
            "topn_score_values": topn_score_values,
            "hub_alpha_values": hub_alpha_values,
        },
        "best": best,
        "trials": trials,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if best:
        print("Best embedding scoring params:", {k: best[k] for k in ["context_n", "topn_score", "hub_alpha"]})
        print("Best metrics:", best["metrics"])
    print("Saved:", args.out)


if __name__ == "__main__":
    main()

