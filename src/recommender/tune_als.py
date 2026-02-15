"""Tune ALS training hyperparameters (re-trains ALS per trial).

This is the "with re-training" tuning: each trial trains an ALS model on TRAIN
only, then evaluates next-POI metrics on the trail protocol.

Search space defaults live in `configs/recommender.toml` under `[tune.als]`.

Note: ALS tuning can get expensive if the search space is large. Keep the number
of evaluated combinations small (defaults are designed to be reasonably fast).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np

from .config import DEFAULT_CONFIG_PATH, load_config
from .models.als import build_interactions, score_als, train_als
from .tune.common import TrailCase, eval_cases, make_train_test_trails, normalize_scores
from .utils_db import get_conn, load_pois, load_visits


def main() -> None:
    p = argparse.ArgumentParser(description="Tune ALS hyperparameters (trail protocol)")
    p.add_argument("--city-qid", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--out", default=os.path.join("data", "reports", "tune_als.json"))

    # Optional overrides (otherwise use [tune.data])
    p.add_argument("--visits-limit", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--min-train", type=int, default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-trials", type=int, default=8, help="Limit number of combinations evaluated")
    args = p.parse_args()

    cfg = load_config(args.config)
    tune = cfg.get("tune", {})
    tune_data = tune.get("data", {})
    tune_a = tune.get("als", {})

    visits_limit = int(args.visits_limit or tune_data.get("visits_limit", 120000))
    k = int(args.k or tune_data.get("k", 20))
    test_size = int(args.test_size or tune_data.get("test_size", 1))
    min_train = int(args.min_train or tune_data.get("min_train", 2))
    max_cases = int(args.max_cases or tune_data.get("max_cases", 150))
    seed = int(args.seed or tune.get("seed", 42))

    factors_values = [int(x) for x in tune_a.get("factors_values", [64, 128])]
    reg_values = [float(x) for x in tune_a.get("regularization_values", [0.01])]
    it_values = [int(x) for x in tune_a.get("iterations_values", [15])]
    alpha_values = [float(x) for x in tune_a.get("alpha_values", [40.0])]

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

    inter, user_to_idx, item_to_idx, idx_to_item = build_interactions(train_df)

    all_combos = list(itertools.product(factors_values, reg_values, it_values, alpha_values))
    rng.shuffle(all_combos)
    combos = all_combos[: max(1, int(args.max_trials))]

    trials: List[Dict[str, object]] = []
    best = None
    best_key = (-1.0, -1.0)

    for factors, reg, iters, alpha in combos:
        model = train_als(inter, factors=int(factors), regularization=float(reg), iterations=int(iters), alpha=float(alpha))

        def _scorer(c: TrailCase) -> Dict[str, float]:
            seen = set([str(x) for x in c.user_items])
            seen.add(str(c.current_poi))
            if c.prev_poi:
                seen.add(str(c.prev_poi))

            al = score_als(model, c.user_id, c.user_items, user_to_idx, item_to_idx, idx_to_item, topn=1200)
            al = {k: v for k, v in al.items() if k in candidate_set and k not in seen}
            return normalize_scores(al)

        m = eval_cases(cases, k=k, scorer_fn=_scorer)
        row = {"factors": factors, "regularization": reg, "iterations": iters, "alpha": alpha, "metrics": m}
        trials.append(row)
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
        "search_space": {
            "factors_values": factors_values,
            "regularization_values": reg_values,
            "iterations_values": it_values,
            "alpha_values": alpha_values,
            "max_trials": int(args.max_trials),
        },
        "best": best,
        "trials": trials,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if best:
        print("Best ALS params:", {k: best[k] for k in ["factors", "regularization", "iterations", "alpha"]})
        print("Best metrics:", best["metrics"])
    print("Saved:", args.out)


if __name__ == "__main__":
    main()

