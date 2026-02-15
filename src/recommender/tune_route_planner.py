"""Tune route planning parameters (no retraining).

Goal:
- Keep next-POI models fixed (scores) and tune how we *select* and *order* an
  itinerary from a candidate pool.

We evaluate using route-level metrics (see `eval/route_metrics.py`) on a subset
of trail cases.

Search space defaults live in `configs/recommender.toml` under `[tune.route_planner]`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config
from .features.cooccurrence import build_cooccurrence
from .features.tfidf import build_tfidf
from .features.transitions import build_transitions, build_transitions_order2
from .models.als import score_als
from .models.content_based import score_content
from .models.co_visitation import score_co_visitation
from .models.embeddings import score_embeddings_context
from .models.markov import next_poi_order2
from .route_planner import plan_route
from .tune.common import (
    TrailCase,
    apply_hub_penalty,
    embedding_context,
    make_train_test_trails,
    normalize_scores,
)
from .utils_db import get_conn, load_poi_categories, load_pois, load_visits
from .eval.route_metrics import compute_route_metrics


def _combine(
    w: Tuple[float, float, float, float, float],
    s_content: Dict[str, float],
    s_item: Dict[str, float],
    s_markov: Dict[str, float],
    s_embed: Dict[str, float],
    s_als: Dict[str, float],
) -> Dict[str, float]:
    wc, wi, wm, we, wa = w
    keys = set(s_content) | set(s_item) | set(s_markov) | set(s_embed) | set(s_als)
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = (
            wc * s_content.get(k, 0.0)
            + wi * s_item.get(k, 0.0)
            + wm * s_markov.get(k, 0.0)
            + we * s_embed.get(k, 0.0)
            + wa * s_als.get(k, 0.0)
        )
    return out


def _objective(route_summary: Dict[str, float]) -> float:
    """Single scalar objective to maximize (heuristic)."""
    # Prefer category alignment + diversity, penalize bad legs.
    return (
        0.45 * float(route_summary.get("cat_match_ratio", 0.0))
        + 0.35 * float(route_summary.get("unique_cat_ratio", 0.0))
        - 0.50 * float(route_summary.get("pct_legs_too_close", 0.0))
        - 0.50 * float(route_summary.get("pct_legs_too_far", 0.0))
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Tune route_planner parameters (route-level metrics)")
    p.add_argument("--city-qid", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--out", default=os.path.join("data", "reports", "tune_route_planner.json"))

    # Optional artifacts (recommended, otherwise hybrid reduces to content/item/markov)
    p.add_argument("--embeddings-path", default=None)
    p.add_argument("--als-path", default=None)

    # Optional overrides (otherwise use [tune.data])
    p.add_argument("--visits-limit", type=int, default=None)
    p.add_argument("--k", type=int, default=8, help="Route length (itinerary size)")
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--min-train", type=int, default=None)
    p.add_argument("--max-cases", type=int, default=120)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-trials", type=int, default=15)
    args = p.parse_args()

    cfg = load_config(args.config)
    tune = cfg.get("tune", {})
    tune_data = tune.get("data", {})
    tune_rp = tune.get("route_planner", {})

    visits_limit = int(args.visits_limit or tune_data.get("visits_limit", 120000))
    test_size = int(args.test_size or tune_data.get("test_size", 1))
    min_train = int(args.min_train or tune_data.get("min_train", 2))
    seed = int(args.seed or tune.get("seed", 42))

    # Fixed hybrid weights for trail-like route building.
    w = tuple(float(x) for x in cfg.get("hybrid", {}).get("trail_current", [0.0, 0.2, 0.65, 0.15, 0.0]))

    rng = random.Random(seed)

    conn = get_conn()
    visits = load_visits(conn, city_qid=args.city_qid, limit=visits_limit)
    pois = load_pois(conn, city_qid=args.city_qid)
    poi_cats = load_poi_categories(conn, fsq_ids=pois["fsq_id"].astype(str).tolist())

    candidate_set = set([str(x) for x in pois["fsq_id"].dropna().astype(str).tolist()])
    visits = visits[visits["venue_id"].astype(str).isin(candidate_set)].copy()
    if visits.empty:
        raise SystemExit("No visits after filtering (check city_qid and data load).")

    train_df, test_df, cases = make_train_test_trails(visits, test_size=test_size, min_train=min_train)
    rng.shuffle(cases)
    cases = cases[: int(args.max_cases)]
    if not cases:
        raise SystemExit("No evaluation cases. Try lowering min_train or increasing visits_limit.")

    # Shared features (TRAIN only).
    tfidf_mat, fsq_ids, _ = build_tfidf(poi_cats)
    co_mat, id_to_idx, idx_to_id = build_cooccurrence(train_df)
    trans1, _ = build_transitions(train_df, pois)
    trans2 = build_transitions_order2(train_df)
    item_counts = train_df["venue_id"].astype(str).value_counts().to_dict()

    # Optional models
    emb_model = joblib.load(args.embeddings_path) if args.embeddings_path else None

    # ALS artifact is stored as a dict: {model, user_to_idx, item_to_idx, idx_to_item}
    als_art = joblib.load(args.als_path) if args.als_path else None
    als_model = als_art.get("model") if isinstance(als_art, dict) else als_art
    user_to_idx: Dict[int, int] = als_art.get("user_to_idx", {}) if isinstance(als_art, dict) else {}
    item_to_idx: Dict[str, int] = als_art.get("item_to_idx", {}) if isinstance(als_art, dict) else {}
    idx_to_item: List[str] = als_art.get("idx_to_item", []) if isinstance(als_art, dict) else []

    emb_cfg = cfg.get("embeddings", {})
    emb_context_n = int(emb_cfg.get("context_n", 3))
    emb_topn = int(emb_cfg.get("topn_score", 2000))
    emb_hub_alpha = float(emb_cfg.get("hub_alpha", 0.0))

    markov_cfg = cfg.get("markov", {})
    markov_hub_alpha = float(markov_cfg.get("hub_alpha", 0.5))

    # Route metric thresholds
    route_cfg = cfg.get("route", {})
    min_leg_km = float(route_cfg.get("min_leg_km", 0.35))
    max_leg_km = float(route_cfg.get("max_leg_km", 5.0))

    # Search space for planner params
    candidate_pool_values = [int(x) for x in tune_rp.get("candidate_pool_values", [300, 600, 1000])]
    pair_min_km_values = [float(x) for x in tune_rp.get("pair_min_km_values", [0.15, 0.22, 0.30])]
    distance_weight_values = [float(x) for x in tune_rp.get("distance_weight_values", [0.2, 0.35, 0.5])]
    diversity_bonus_values = [float(x) for x in tune_rp.get("diversity_bonus_values", [0.0, 0.06, 0.1])]
    max_per_category_values = [int(x) for x in tune_rp.get("max_per_category_values", [1, 2, 3])]

    combos = list(
        itertools.product(
            candidate_pool_values,
            pair_min_km_values,
            distance_weight_values,
            diversity_bonus_values,
            max_per_category_values,
        )
    )
    rng.shuffle(combos)
    combos = combos[: max(1, int(args.max_trials))]

    trials: List[Dict[str, object]] = []
    best = None
    best_obj = -1e18

    # Build fast POI lookup for candidate materialization.
    pois_idx = pois.copy()
    pois_idx["fsq_id"] = pois_idx["fsq_id"].astype(str)
    pois_idx = pois_idx.set_index("fsq_id", drop=False)
    pois_primary = pois_idx["primary_category"].to_dict() if "primary_category" in pois_idx.columns else {}

    # Precompute combined candidate lists per case once (fixed model scores + fixed hybrid weights).
    max_pool = max(candidate_pool_values) if candidate_pool_values else 600
    per_case: List[Dict[str, object]] = []
    for c in cases:
        seen = set([str(x) for x in c.user_items])
        seen.add(str(c.current_poi))
        if c.prev_poi:
            seen.add(str(c.prev_poi))

        s_content = normalize_scores(
            {k: v for k, v in score_content(c.user_items, fsq_ids, tfidf_mat).items() if k in candidate_set and k not in seen}
        )
        s_item = normalize_scores(
            {k: v for k, v in score_co_visitation(c.user_items, co_mat, id_to_idx, idx_to_id).items() if k in candidate_set and k not in seen}
        )
        mk = dict(next_poi_order2(c.prev_poi, c.current_poi, trans2, trans1, topn=200, backoff=0.35))
        mk = {k: v for k, v in mk.items() if k in candidate_set and k not in seen}
        s_markov = normalize_scores(apply_hub_penalty(mk, item_counts, markov_hub_alpha))

        s_embed: Dict[str, float] = {}
        if emb_model is not None:
            ctx = embedding_context(c.user_items, c.current_poi, context_n=emb_context_n)
            eb = score_embeddings_context(emb_model, ctx, topn=min(int(emb_topn), 5000))
            eb = {k: v for k, v in eb.items() if k in candidate_set and k not in seen}
            s_embed = normalize_scores(apply_hub_penalty(eb, item_counts, emb_hub_alpha))

        s_als: Dict[str, float] = {}
        if als_model is not None and user_to_idx and item_to_idx:
            al = score_als(als_model, c.user_id, c.user_items, user_to_idx, item_to_idx, idx_to_item, topn=1200)
            al = {k: v for k, v in al.items() if k in candidate_set and k not in seen}
            s_als = normalize_scores(al)

        comb = _combine(w, s_content, s_item, s_markov, s_embed, s_als)
        top = sorted(comb.items(), key=lambda t: t[1], reverse=True)[: int(max_pool)]
        if not top:
            continue
        ids = [x for x, _ in top]
        scores = [float(s) for _, s in top]
        per_case.append({"case": c, "ids": ids, "scores": scores})

    for candidate_pool, pair_min_km, dist_w, div_bonus, max_per_cat in combos:
        per_route_metrics: List[Dict[str, float]] = []

        for entry in per_case:
            c: TrailCase = entry["case"]  # type: ignore[assignment]
            ids: List[str] = entry["ids"]  # type: ignore[assignment]
            scores: List[float] = entry["scores"]  # type: ignore[assignment]

            sub_ids = ids[: int(candidate_pool)]
            sub_scores = scores[: int(candidate_pool)]
            if not sub_ids:
                continue

            cand_df = pois_idx.reindex(sub_ids)
            cand_df = cand_df.dropna(subset=["lat", "lon"])
            if cand_df.empty:
                continue
            cand_df = cand_df.copy()
            # Align scores with remaining ids after dropping NaNs.
            score_map = {pid: float(s) for pid, s in zip(sub_ids, sub_scores)}
            cand_df["score"] = cand_df["fsq_id"].astype(str).map(score_map).astype(float)
            cand_df = cand_df.dropna(subset=["score"])
            if cand_df.empty:
                continue

            planned = plan_route(
                cand_df,
                k=int(args.k),
                anchor=None,  # route planner tuning is independent of location here
                min_leg_km=min_leg_km,
                max_leg_km=max_leg_km,
                pair_min_km=float(pair_min_km),
                max_per_category=int(max_per_cat),
                distance_weight=float(dist_w),
                diversity_bonus=float(div_bonus),
            )
            if planned.ordered_df.empty:
                continue

            # Build "user preference categories" from history (primary_category only; fast).
            user_cats = [pois_primary.get(str(x)) for x in c.user_items]
            user_cats = [str(x) for x in user_cats if x and str(x) != "nan"]
            user_top = []
            if user_cats:
                # top 5
                counts = {}
                for cc in user_cats:
                    counts[cc] = counts.get(cc, 0) + 1
                user_top = [x for x, _ in sorted(counts.items(), key=lambda t: t[1], reverse=True)[:5]]

            m = compute_route_metrics(
                planned.ordered_df,
                total_km=float(planned.total_km),
                anchor=None,
                preferred_categories=user_top,
                min_leg_km=min_leg_km,
                max_leg_km=max_leg_km,
            ).to_dict()
            per_route_metrics.append(m)

        if not per_route_metrics:
            continue

        dfm = pd.DataFrame(per_route_metrics)
        summary = {k: float(dfm[k].mean()) for k in dfm.columns}
        obj = _objective(summary)
        row = {
            "candidate_pool": int(candidate_pool),
            "pair_min_km": float(pair_min_km),
            "distance_weight": float(dist_w),
            "diversity_bonus": float(div_bonus),
            "max_per_category": int(max_per_cat),
            "objective": float(obj),
            "summary": summary,
        }
        trials.append(row)
        if obj > best_obj:
            best_obj = obj
            best = row

    out = {
        "city_qid": args.city_qid,
        "k": int(args.k),
        "visits_limit": visits_limit,
        "test_size": test_size,
        "min_train": min_train,
        "max_cases": int(args.max_cases),
        "weights_used": list(w),
        "search_space": {
            "candidate_pool_values": candidate_pool_values,
            "pair_min_km_values": pair_min_km_values,
            "distance_weight_values": distance_weight_values,
            "diversity_bonus_values": diversity_bonus_values,
            "max_per_category_values": max_per_category_values,
            "max_trials": int(args.max_trials),
        },
        "best": best,
        "trials": trials,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if best:
        print("Best route_planner params:", {k: best[k] for k in ["candidate_pool", "pair_min_km", "distance_weight", "diversity_bonus", "max_per_category"]})
        print("Best summary:", best["summary"])
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
