"""
Route-level evaluation: evaluate the *itinerary* produced by the recommender.

This complements classic next-POI metrics (Hit/MRR/NDCG) by measuring:
- spatial coherence (legs not too short/too long, total distance)
- diversity (unique categories, entropy)
- alignment with a user profile (category match ratio)

Important: this evaluator is optimized to be usable in practice:
- it precomputes global features once (TF-IDF, co-occurrence, transitions)
- it loads (or trains, if --fair) embedding/ALS models once
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONFIG_PATH, load_config
from ..features.cooccurrence import build_cooccurrence
from ..features.tfidf import build_tfidf
from ..features.transitions import build_transitions, build_transitions_order2
from ..features.word2vec import sequences_from_visits
from ..models.als import build_interactions, score_als, train_als
from ..models.content_based import score_content
from ..models.co_visitation import score_co_visitation
from ..models.embeddings import score_embeddings_context, score_embeddings_next, train_embeddings
from ..models.markov import next_poi_order2
from ..route_planner import plan_route
from ..utils_db import get_conn, load_poi_categories, load_pois, load_visits
from .evaluate import split_train_test, split_train_test_last_trail_user, split_train_test_trails
from .route_metrics import compute_route_metrics


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float32)
    vmax = float(vals.max())
    if vmax <= 0:
        return {}
    return {k: float(v) / vmax for k, v in scores.items()}


def _user_top_categories(
    user_items: Sequence[str],
    pois: pd.DataFrame,
    poi_cats: pd.DataFrame,
    topn: int = 5,
) -> List[str]:
    if not user_items:
        return []
    cat_map = {}
    if not pois.empty and "fsq_id" in pois.columns and "primary_category" in pois.columns:
        cat_map = pois.set_index("fsq_id")["primary_category"].to_dict()
    cats = [cat_map.get(x) for x in user_items]
    cats = [str(c) for c in cats if c and str(c) != "nan"]
    if cats:
        return [c for c, _ in Counter(cats).most_common(topn)]
    if poi_cats.empty:
        return []
    sub = poi_cats[poi_cats["fsq_id"].astype(str).isin([str(x) for x in user_items])]
    if sub.empty:
        return []
    names = sub["category_name"].astype(str).tolist()
    return [c for c, _ in Counter(names).most_common(topn)]


def _anchor_from_poi(current_poi: Optional[str], pois: pd.DataFrame) -> Optional[Tuple[float, float]]:
    if not current_poi or pois.empty:
        return None
    row = pois[pois["fsq_id"].astype(str) == str(current_poi)]
    if row.empty:
        return None
    try:
        return float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])
    except Exception:
        return None


def _apply_hub_penalty(scores: Dict[str, float], item_counts: Dict[str, int], alpha: float) -> Dict[str, float]:
    """
    Penalize extremely popular items (hubs) to reduce generic recommendations.

    score' = score / (count ** alpha)
    - alpha=0 -> no change
    """
    if not scores:
        return {}
    if alpha <= 0:
        return scores
    out: Dict[str, float] = {}
    for k, v in scores.items():
        c = float(item_counts.get(str(k), 1))
        out[str(k)] = float(v) / (c**alpha if c > 0 else 1.0)
    return out


def _embedding_context(user_items: Sequence[str], current_poi: Optional[str], context_n: int) -> List[str]:
    ctx = [str(x) for x in user_items if x]
    if current_poi:
        # Ensure current_poi is part of the context and keep it as the most recent.
        ctx = [x for x in ctx if x != str(current_poi)]
        ctx.append(str(current_poi))
    if context_n <= 0:
        return ctx[-1:] if ctx else []
    return ctx[-context_n:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Route-level evaluation (itinerary quality)")
    parser.add_argument("--dsn", help="Postgres DSN (otherwise POSTGRES_DSN/DEFAULT)")
    parser.add_argument("--city", help="Optional filter over pois.city")
    parser.add_argument("--city-qid", dest="city_qid", help="Filter over visits.venue_city and pois.city_qid")
    parser.add_argument("--protocol", choices=["user", "trail", "last_trail_user"], default="trail")
    parser.add_argument("--k", type=int, default=10, help="Route length (number of POIs)")
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--min-train", type=int, default=2)
    parser.add_argument("--min-test-pois", type=int, default=4, help="Mínimo de POIs en test para last_trail_user")
    parser.add_argument("--max-cases", type=int, default=200, help="Max users/trails to evaluate")
    parser.add_argument("--visits-limit", type=int, help="Limit visits loaded (speed)")
    parser.add_argument("--seed", type=int, help="Semilla para muestreo reproducible (cuando hay límites)")
    parser.add_argument("--modes", nargs="+", default=["hybrid", "markov", "embed", "item", "content", "als"])
    parser.add_argument("--use-embeddings", action="store_true")
    parser.add_argument("--embeddings-path", default="src/recommender/cache/word2vec.joblib")
    parser.add_argument("--use-als", action="store_true")
    parser.add_argument("--als-path", default="src/recommender/cache/als_model.joblib")
    parser.add_argument("--fair", action="store_true", help="Train models on train split (slower, leak-free)")
    parser.add_argument("--output", help="Write JSON metrics to a file")
    args = parser.parse_args()

    cfg = load_config(DEFAULT_CONFIG_PATH, city_qid=args.city_qid)
    hyb_cfg = cfg.get("hybrid", {})
    emb_cfg = cfg.get("embeddings", {})
    als_cfg = cfg.get("als", {})
    route_cfg = cfg.get("route", {})
    route_pl_cfg = cfg.get("route_planner", {})
    eval_cfg = cfg.get("eval", {})
    eval_routes_cfg = cfg.get("eval_routes", {})
    markov_cfg = cfg.get("markov", {})
    exclude_categories = set(cfg.get("filters", {}).get("exclude_categories", ["Intersection", "State", "Home (private)"]))

    min_leg_km = float(route_cfg.get("min_leg_km", 0.3))
    max_leg_km = float(route_cfg.get("max_leg_km", 5.0))
    user_topn = int(eval_routes_cfg.get("user_top_categories", 5))

    conn = get_conn(args.dsn)
    visits = load_visits(conn, city_qid=args.city_qid, limit=args.visits_limit)
    pois = load_pois(conn, city=args.city, city_qid=args.city_qid)
    poi_cats = load_poi_categories(conn, fsq_ids=pois["fsq_id"]) if not pois.empty else pd.DataFrame()

    if args.protocol == "trail":
        train_visits, test_visits = split_train_test_trails(visits, test_size=args.test_size, min_train=args.min_train)
        case_ids = list(test_visits["trail_id"].unique())
    elif args.protocol == "last_trail_user":
        train_visits, test_visits = split_train_test_last_trail_user(
            visits,
            min_train=args.min_train,
            min_test_pois=args.min_test_pois,
        )
        case_ids = list(test_visits["user_id"].unique())
    else:
        train_visits, test_visits = split_train_test(visits, test_size=args.test_size, min_train=args.min_train)
        case_ids = list(test_visits["user_id"].unique())

    seed = args.seed if args.seed is not None else eval_routes_cfg.get("seed", eval_cfg.get("seed"))

    # Make the evaluated subset reproducible when limiting cases.
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(case_ids)
    max_cases = int(args.max_cases) if args.max_cases else None

    if train_visits.empty or test_visits.empty or pois.empty:
        raise SystemExit("Not enough data for route evaluation (adjust filters/split).")

    # Global features (computed once)
    tfidf_matrix, tfidf_ids, _ = build_tfidf(poi_cats)
    co_mat, id_to_idx, idx_to_id = build_cooccurrence(train_visits)
    trans_poi1, _ = build_transitions(train_visits, pois)
    trans_poi2 = build_transitions_order2(train_visits)

    # Load / train embedding model once
    embed_model = None
    if args.use_embeddings:
        if (not args.fair) and args.embeddings_path:
            try:
                import joblib

                embed_model = joblib.load(args.embeddings_path)
            except Exception:
                embed_model = None
        if embed_model is None:
            seqs = sequences_from_visits(train_visits)
            # Light params under fair to keep runtime down.
            e_vector_size = int(eval_cfg.get("emb_vector_size", emb_cfg.get("vector_size", 128))) if args.fair else int(emb_cfg.get("vector_size", 128))
            e_window = int(eval_cfg.get("emb_window", emb_cfg.get("window", 15))) if args.fair else int(emb_cfg.get("window", 15))
            e_epochs = int(eval_cfg.get("emb_epochs", emb_cfg.get("epochs", 10))) if args.fair else int(emb_cfg.get("epochs", 10))
            e_negative = int(eval_cfg.get("emb_negative", emb_cfg.get("negative", 5))) if args.fair else int(emb_cfg.get("negative", 5))
            embed_model = train_embeddings(
                seqs,
                vector_size=e_vector_size,
                window=e_window,
                min_count=int(emb_cfg.get("min_count", 2)),
                workers=int(emb_cfg.get("workers", 4)),
                epochs=e_epochs,
                negative=e_negative,
                sample=float(emb_cfg.get("sample", 1e-3)),
                ns_exponent=float(emb_cfg.get("ns_exponent", 0.75)),
                hs=int(emb_cfg.get("hs", 0)),
                seed=int(emb_cfg.get("seed", 42)),
            )
            if args.embeddings_path:
                try:
                    import joblib

                    joblib.dump(embed_model, args.embeddings_path)
                except Exception:
                    pass

    # Load / train ALS once
    als_obj = None
    if args.use_als:
        if (not args.fair) and args.als_path:
            try:
                import joblib

                als_obj = joblib.load(args.als_path)
            except Exception:
                als_obj = None
        if als_obj is None:
            interactions, user_to_idx, item_to_idx, idx_to_item = build_interactions(train_visits)
            a_factors = int(eval_cfg.get("als_factors", als_cfg.get("factors", 64))) if args.fair else int(als_cfg.get("factors", 64))
            a_reg = float(eval_cfg.get("als_regularization", als_cfg.get("regularization", 0.01))) if args.fair else float(als_cfg.get("regularization", 0.01))
            a_iters = int(eval_cfg.get("als_iterations", als_cfg.get("iterations", 15))) if args.fair else int(als_cfg.get("iterations", 15))
            a_alpha = float(eval_cfg.get("als_alpha", als_cfg.get("alpha", 40.0))) if args.fair else float(als_cfg.get("alpha", 40.0))
            als_model = train_als(interactions, factors=a_factors, regularization=a_reg, iterations=a_iters, alpha=a_alpha)
            als_obj = {"model": als_model, "user_to_idx": user_to_idx, "item_to_idx": item_to_idx, "idx_to_item": idx_to_item}
            if args.als_path:
                try:
                    import joblib

                    joblib.dump(als_obj, args.als_path)
                except Exception:
                    pass

    allowed_ids = set(pois["fsq_id"].astype(str))
    # Popularity counts for hub penalty (computed on train split).
    item_counts: Dict[str, int] = train_visits["venue_id"].astype(str).value_counts().to_dict()

    per_route = defaultdict(list)
    skipped_truth = 0
    n_cases_used = 0

    for cid in case_ids:
        if max_cases is not None and n_cases_used >= max_cases:
            break
        # Determine current/prev POI + "seen" set for sequential context.
        if args.protocol == "trail":
            trail_train = train_visits[train_visits["trail_id"] == cid].sort_values("timestamp")
            trail_test = test_visits[test_visits["trail_id"] == cid].sort_values("timestamp")
            if trail_train.empty or trail_test.empty:
                continue
            uid = int(trail_train.iloc[-1]["user_id"])
            seq_items = trail_train["venue_id"].astype(str).tolist()
            seen = set(seq_items)
            truth_items = [t for t in trail_test["venue_id"].astype(str).tolist() if t not in seen]
            if not truth_items:
                skipped_truth += 1
                continue
            current_poi = str(seq_items[-1])
            prev_poi = str(seq_items[-2]) if len(seq_items) >= 2 else None
        elif args.protocol == "last_trail_user":
            uid = int(cid)
            user_train = train_visits[train_visits["user_id"] == uid].sort_values("timestamp")
            user_test = test_visits[test_visits["user_id"] == uid].sort_values("timestamp")
            if user_train.empty or user_test.empty:
                continue
            user_items_seq = user_train["venue_id"].astype(str).tolist()
            test_seq = user_test["venue_id"].astype(str).tolist()
            if len(test_seq) < 2:
                continue
            # Seed = first POI of the held-out last trail.
            current_poi = str(test_seq[0])
            prev_poi = None
            seen = set(user_items_seq)
            seen.add(current_poi)
            truth_items = [t for t in test_seq[1:] if t not in seen]
            if not truth_items:
                skipped_truth += 1
                continue
        else:
            uid = int(cid)
            user_train = train_visits[train_visits["user_id"] == uid].sort_values("timestamp")
            user_test = test_visits[test_visits["user_id"] == uid].sort_values("timestamp")
            if user_train.empty or user_test.empty:
                continue
            user_items_seq = user_train["venue_id"].astype(str).tolist()
            seen = set(user_items_seq)
            truth_items = [t for t in user_test["venue_id"].astype(str).tolist() if t not in seen]
            if not truth_items:
                skipped_truth += 1
                continue
            current_poi = str(user_items_seq[-1])
            prev_poi = str(user_items_seq[-2]) if len(user_items_seq) >= 2 else None

        user_items = train_visits[train_visits["user_id"] == uid]["venue_id"].astype(str).tolist()
        pref_cats = _user_top_categories(user_items, pois=pois, poi_cats=poi_cats, topn=user_topn)
        anchor = _anchor_from_poi(current_poi, pois)

        # Compute per-engine scores (once per case)
        content_scores = _normalize_scores(score_content(user_items, tfidf_ids, tfidf_matrix))
        co_scores = _normalize_scores(score_co_visitation(user_items, co_mat, id_to_idx, idx_to_id))
        # Markov (order-2 with backoff) + optional hub penalty.
        markov_raw = dict(next_poi_order2(prev_poi, current_poi, trans_poi2, trans_poi1, topn=2000, backoff=0.3))
        markov_raw = _apply_hub_penalty(markov_raw, item_counts=item_counts, alpha=float(markov_cfg.get("hub_alpha", 0.0)))
        markov_scores = _normalize_scores(markov_raw)

        # Embeddings: prefer context (last-N items) instead of only current POI.
        embed_scores: Dict[str, float] = {}
        if embed_model is not None:
            ctx_n = int(emb_cfg.get("context_n", 1))
            ctx = _embedding_context(user_items=user_items, current_poi=current_poi, context_n=ctx_n)
            topn = int(emb_cfg.get("topn_score", 1000))
            if len(ctx) >= 2:
                emb_raw = score_embeddings_context(embed_model, ctx, topn=topn)
            else:
                emb_raw = score_embeddings_next(embed_model, current_poi, topn=topn)
            emb_raw = _apply_hub_penalty(emb_raw, item_counts=item_counts, alpha=float(emb_cfg.get("hub_alpha", 0.0)))
            embed_scores = _normalize_scores(emb_raw)
        als_scores = {}
        if als_obj is not None:
            als_scores = _normalize_scores(
                score_als(
                    als_obj["model"],
                    uid,
                    user_items,
                    als_obj["user_to_idx"],
                    als_obj["item_to_idx"],
                    als_obj["idx_to_item"],
                    topn=int(als_cfg.get("topn_score", 500)),
                )
            )

        # Restrict to allowed POIs
        content_scores = {k: v for k, v in content_scores.items() if k in allowed_ids and k not in seen}
        co_scores = {k: v for k, v in co_scores.items() if k in allowed_ids and k not in seen}
        markov_scores = {k: v for k, v in markov_scores.items() if k in allowed_ids and k not in seen}
        embed_scores = {k: v for k, v in embed_scores.items() if k in allowed_ids and k not in seen}
        als_scores = {k: v for k, v in als_scores.items() if k in allowed_ids and k not in seen}

        def combine(weights: Sequence[float]) -> Dict[str, float]:
            w_content, w_item, w_markov, w_embed, w_als = weights
            all_ids = set(content_scores) | set(co_scores) | set(markov_scores) | set(embed_scores) | set(als_scores)
            out: Dict[str, float] = {}
            for fid in all_ids:
                out[fid] = (
                    w_content * content_scores.get(fid, 0.0)
                    + w_item * co_scores.get(fid, 0.0)
                    + w_markov * markov_scores.get(fid, 0.0)
                    + w_embed * embed_scores.get(fid, 0.0)
                    + w_als * als_scores.get(fid, 0.0)
                )
            return out

        # Evaluate each mode -> generate a route -> compute route metrics
        for mode in args.modes:
            if mode == "content":
                scores = content_scores
            elif mode == "item":
                scores = co_scores
            elif mode == "markov":
                scores = markov_scores
            elif mode == "embed":
                scores = embed_scores
            elif mode == "als":
                scores = als_scores
            else:  # hybrid
                # Use the same weight selection logic as scorer (roughly).
                # If we're evaluating a trail protocol, bias towards sequential signals.
                if args.protocol == "trail" and current_poi:
                    w = list(hyb_cfg.get("trail_current", [0.05, 0.10, 0.55, 0.30, 0.00]))
                elif user_items and current_poi:
                    w = list(hyb_cfg.get("user_current", [0.10, 0.15, 0.15, 0.05, 0.55]))
                elif user_items:
                    w = list(hyb_cfg.get("user_only", [0.10, 0.20, 0.10, 0.05, 0.55]))
                else:
                    w = list(hyb_cfg.get("cold_start", [0.60, 0.20, 0.20, 0.00, 0.00]))
                if not als_scores:
                    w[4] = 0.0
                if not embed_scores:
                    w[3] = 0.0
                scores = combine(w)

            if not scores:
                continue

            # Candidate table + basic filters
            cand = (
                pd.DataFrame({"fsq_id": list(scores.keys()), "score": list(scores.values())})
                .merge(pois, on="fsq_id", how="left")
                .dropna(subset=["name", "primary_category", "lat", "lon"], how="any")
            )
            if cand.empty:
                continue
            cand = cand[~cand["primary_category"].isin(exclude_categories)]
            cand = cand.sort_values("score", ascending=False)

            # Route-aware selection: pick K POIs with distance constraints + soft diversity.
            pool = int(route_pl_cfg.get("candidate_pool", max(500, int(args.k) * 50)))
            planned = plan_route(
                cand.head(pool),
                k=int(args.k),
                anchor=anchor,
                min_leg_km=float(min_leg_km),
                max_leg_km=float(max_leg_km),
                pair_min_km=float(route_pl_cfg.get("pair_min_km", 0.2)),
                max_per_category=int(route_pl_cfg.get("max_per_category", 2)),
                distance_weight=float(route_pl_cfg.get("distance_weight", 0.35)),
                distance_weight_no_anchor=(
                    float(route_pl_cfg["distance_weight_no_anchor"]) if route_pl_cfg.get("distance_weight_no_anchor") is not None else None
                ),
                max_leg_km_no_anchor=(
                    float(route_pl_cfg["max_leg_km_no_anchor"]) if route_pl_cfg.get("max_leg_km_no_anchor") is not None else None
                ),
                diversity_bonus=float(route_pl_cfg.get("diversity_bonus", 0.05)),
            )
            if planned.ordered_df.empty:
                continue
            m = compute_route_metrics(
                planned.ordered_df,
                total_km=planned.total_km,
                anchor=anchor,
                preferred_categories=pref_cats,
                min_leg_km=min_leg_km,
                max_leg_km=max_leg_km,
            )
            per_route[mode].append(m.to_dict())

        n_cases_used += 1

    # Aggregate (mean) per mode
    agg_rows = []
    for mode, rows in per_route.items():
        dfm = pd.DataFrame(rows)
        if dfm.empty:
            continue
        mean = dfm.mean(numeric_only=True).to_dict()
        mean["mode"] = mode
        mean["n_routes"] = int(len(dfm))
        agg_rows.append(mean)

    out = {
        "summary": {
            "protocol": args.protocol,
            "k": int(args.k),
            "min_leg_km": min_leg_km,
            "max_leg_km": max_leg_km,
            "n_cases_used": n_cases_used,
            "cases_skipped_all_truth_seen": int(skipped_truth),
        },
        "aggregate": agg_rows,
        "per_route": per_route,
    }

    if agg_rows:
        df_agg = pd.DataFrame(agg_rows).set_index("mode").sort_values("n_routes", ascending=False)
        with pd.option_context("display.max_columns", None, "display.width", 140):
            print(df_agg[["n_routes", "total_km", "avg_leg_km", "pct_legs_too_close", "pct_legs_too_far", "unique_cat_ratio", "cat_entropy", "cat_match_ratio"]])
    else:
        print("No route metrics computed (no routes).")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved route evaluation to {args.output}")


if __name__ == "__main__":
    main()
