"""Tune Markov scoring parameters (no model retraining).

What we tune:
- order-2 backoff to order-1
- hub penalty alpha (penalize extremely popular next-POIs)

Defaults/search space live in `configs/recommender.toml` under `[tune.markov]`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config
from .features.transitions import build_transitions, build_transitions_order2
from .models.markov import next_poi_order2
from .tune.common import TrailCase, apply_hub_penalty, eval_cases, make_train_test_trails, normalize_scores
from .utils_db import get_conn, load_pois, load_visits


def main() -> None:
    p = argparse.ArgumentParser(description="Tune Markov parameters (trail protocol)")
    p.add_argument("--city-qid", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--out", default=os.path.join("data", "reports", "tune_markov.json"))

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
    tune_m = tune.get("markov", {})

    visits_limit = int(args.visits_limit or tune_data.get("visits_limit", 120000))
    k = int(args.k or tune_data.get("k", 20))
    test_size = int(args.test_size or tune_data.get("test_size", 1))
    min_train = int(args.min_train or tune_data.get("min_train", 2))
    max_cases = int(args.max_cases or tune_data.get("max_cases", 150))
    seed = int(args.seed or tune.get("seed", 42))

    backoff_values = [float(x) for x in tune_m.get("backoff_values", [0.3, 0.35, 0.45])]
    hub_alpha_values = [float(x) for x in tune_m.get("hub_alpha_values", [0.0, 0.5])]
    topn = int(tune_m.get("topn", 150))

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

    trans1, _ = build_transitions(train_df, pois)
    trans2 = build_transitions_order2(train_df)
    item_counts = train_df["venue_id"].astype(str).value_counts().to_dict()

    results: List[Dict[str, object]] = []

    best = None
    best_key = (-1.0, -1.0)
    for backoff in backoff_values:
        for hub_alpha in hub_alpha_values:

            def _scorer(c: TrailCase) -> Dict[str, float]:
                seen = set([str(x) for x in c.user_items])
                seen.add(str(c.current_poi))
                if c.prev_poi:
                    seen.add(str(c.prev_poi))

                mk = dict(next_poi_order2(c.prev_poi, c.current_poi, trans2, trans1, topn=topn, backoff=backoff))
                mk = {k: v for k, v in mk.items() if k in candidate_set and k not in seen}
                mk = normalize_scores(apply_hub_penalty(mk, item_counts, hub_alpha))
                return mk

            m = eval_cases(cases, k=k, scorer_fn=_scorer)
            row = {"backoff": backoff, "hub_alpha": hub_alpha, "metrics": m}
            results.append(row)
            key = (m["ndcg"], m["mrr"])
            if key > best_key:
                best_key = key
                best = row

    out = {
        "city_qid": args.city_qid,
        "k": k,
        "visits_limit": visits_limit,
        "test_size": test_size,
        "min_train": min_train,
        "max_cases": max_cases,
        "search_space": {"backoff_values": backoff_values, "hub_alpha_values": hub_alpha_values, "topn": topn},
        "best": best,
        "trials": results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if best:
        print("Best Markov params:", {"backoff": best["backoff"], "hub_alpha": best["hub_alpha"]})
        print("Best metrics:", best["metrics"])
    print("Saved:", args.out)


if __name__ == "__main__":
    main()

