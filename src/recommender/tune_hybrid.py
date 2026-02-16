"""Quick tuner for hybrid weights (fast, <~5 min on typical laptop).

This script tunes `hybrid.trail_current` weights on a single city using an
offline *trail* protocol (predict next POI given the previous/current POI).

Why: hybrid weights are the easiest high-impact knob to improve results without
retraining all models. We keep the CLI minimal; tuned weights can be copied into
`configs/recommender.toml`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config
from .eval.evaluate import compute_metrics, split_train_test_trails
from .features.cooccurrence import build_cooccurrence
from .features.tfidf import build_tfidf
from .features.transitions import build_transitions, build_transitions_order2
from .models.als import build_interactions, score_als, train_als
from .models.content_based import score_content
from .models.co_visitation import score_co_visitation
from .models.embeddings import score_embeddings_context
from .models.markov import next_poi_order2
from .utils_db import get_conn, load_poi_categories, load_pois, load_visits


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vmax = max(scores.values())
    if vmax <= 0:
        return {}
    return {k: float(v) / float(vmax) for k, v in scores.items()}


def _apply_hub_penalty(scores: Dict[str, float], item_counts: Dict[str, int], alpha: float) -> Dict[str, float]:
    if not scores:
        return {}
    if alpha <= 0:
        return scores
    out: Dict[str, float] = {}
    for k, v in scores.items():
        c = float(item_counts.get(str(k), 1))
        out[str(k)] = float(v) / (c**alpha if c > 0 else 1.0)
    return out


def _embedding_context(user_items: List[str], current_poi: str, context_n: int) -> List[str]:
    ctx = [str(x) for x in user_items if x]
    # Keep current_poi as the most recent item.
    ctx = [x for x in ctx if x != str(current_poi)] + [str(current_poi)]
    if context_n <= 0:
        return ctx[-1:]
    return ctx[-context_n:]


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


def _sample_weights(rng: random.Random, base: Sequence[float], jitter: float = 0.10) -> Tuple[float, float, float, float, float]:
    """Sample around a base weight vector with small jitter, then renormalize."""
    x = []
    for v in base:
        dv = rng.uniform(-jitter, jitter)
        x.append(max(0.0, float(v) + dv))
    s = sum(x) or 1.0
    x = [v / s for v in x]
    return (x[0], x[1], x[2], x[3], x[4])


@dataclass
class Case:
    user_id: int
    prev_poi: Optional[str]
    current_poi: str
    user_items: List[str]
    truth: List[str]


def build_cases(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[Case]:
    """Trail protocol cases: for each trail, use last 1-2 POIs from train as context, test as truth."""
    if train_df.empty or test_df.empty:
        return []

    user_hist = train_df.sort_values("timestamp").groupby("user_id")["venue_id"].apply(lambda s: [str(x) for x in s.tolist()])

    cases: List[Case] = []
    for tid, g_test in test_df.groupby("trail_id"):
        g_train = train_df[train_df["trail_id"] == tid].sort_values("timestamp")
        if g_train.empty:
            continue
        seq = [str(x) for x in g_train["venue_id"].tolist()]
        cur = seq[-1]
        prev = seq[-2] if len(seq) >= 2 else None
        uid = int(g_train["user_id"].iloc[0])
        truth = [str(x) for x in g_test.sort_values("timestamp")["venue_id"].tolist()]
        items = user_hist.get(uid, [])
        cases.append(Case(user_id=uid, prev_poi=prev, current_poi=cur, user_items=list(items), truth=truth))
    return cases


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city-qid", required=True, help="City QID (e.g. Q35765)")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config TOML path")
    p.add_argument("--visits-limit", type=int, default=120000)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--test-size", type=int, default=1)
    p.add_argument("--min-train", type=int, default=2)
    p.add_argument("--max-cases", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)

    # Optional artifacts
    p.add_argument("--use-embeddings", action="store_true")
    p.add_argument("--embeddings-path", default=None)
    p.add_argument("--use-als", action="store_true")
    p.add_argument("--als-path", default=None)

    p.add_argument("--trials", type=int, default=25, help="How many weight candidates to try")
    p.add_argument("--out", default=os.path.join("data", "reports", "tune_hybrid.json"))
    args = p.parse_args()

    cfg = load_config(args.config, city_qid=args.city_qid)
    rng = random.Random(int(args.seed))

    conn = get_conn()
    visits = load_visits(conn, city_qid=args.city_qid, limit=int(args.visits_limit))
    pois = load_pois(conn, city_qid=args.city_qid)
    poi_cats = load_poi_categories(conn, fsq_ids=pois["fsq_id"].astype(str).tolist())

    # Keep only POIs we can recommend (intersection of visits and pois).
    candidate_set = set([str(x) for x in pois["fsq_id"].dropna().astype(str).tolist()])
    visits = visits[visits["venue_id"].astype(str).isin(candidate_set)].copy()
    if visits.empty or len(visits) < 1000:
        raise SystemExit("Not enough visits after filtering by city_qid and POIs.")

    train_df, test_df = split_train_test_trails(visits, test_size=int(args.test_size), min_train=int(args.min_train))
    cases = build_cases(train_df, test_df)
    rng.shuffle(cases)
    cases = cases[: int(args.max_cases)]
    if not cases:
        raise SystemExit("No evaluation cases (try lowering --min-train or increasing --visits-limit).")

    # Build shared features/models from TRAIN only (avoid leakage).
    tfidf_mat, fsq_ids, _ = build_tfidf(poi_cats)
    co_mat, id_to_idx, idx_to_id = build_cooccurrence(train_df)
    trans1, _ = build_transitions(train_df, pois)
    trans2 = build_transitions_order2(train_df)

    # Popularity counts (for hub penalty)
    item_counts = train_df["venue_id"].astype(str).value_counts().to_dict()

    # Embeddings
    emb_model = None
    if args.use_embeddings:
        if not args.embeddings_path:
            raise SystemExit("--use-embeddings requiere --embeddings-path")
        emb_model = joblib.load(args.embeddings_path)

    # ALS
    als_model = None
    user_to_idx: Dict[int, int] = {}
    item_to_idx: Dict[str, int] = {}
    idx_to_item: List[str] = []
    if args.use_als:
        inter, user_to_idx, item_to_idx, idx_to_item = build_interactions(train_df)
        als_cfg = cfg.get("eval", {})
        als_model = train_als(
            inter,
            factors=int(als_cfg.get("als_factors", 64)),
            regularization=float(als_cfg.get("als_regularization", 0.01)),
            iterations=int(als_cfg.get("als_iterations", 20)),
            alpha=float(als_cfg.get("als_alpha", 40.0)),
        )

    # Candidate weight vectors: include base + random jitters around it.
    base = cfg.get("hybrid", {}).get("trail_current", [0.0, 0.2, 0.65, 0.15, 0.0])
    candidates: List[Tuple[float, float, float, float, float]] = []
    candidates.append(tuple(float(x) for x in base))  # type: ignore[arg-type]
    for _ in range(max(0, int(args.trials) - 1)):
        candidates.append(_sample_weights(rng, base, jitter=0.12))

    emb_cfg = cfg.get("embeddings", {})
    emb_context_n = int(emb_cfg.get("context_n", 3))
    emb_topn = int(emb_cfg.get("topn_score", 2000))
    emb_hub_alpha = float(emb_cfg.get("hub_alpha", 0.0))

    markov_cfg = cfg.get("markov", {})
    markov_hub_alpha = float(markov_cfg.get("hub_alpha", 0.5))

    def _eval_weights(w: Tuple[float, float, float, float, float]) -> Dict[str, float]:
        mets = []
        for c in cases:
            seen = set([str(x) for x in c.user_items])
            seen.add(str(c.current_poi))
            if c.prev_poi:
                seen.add(str(c.prev_poi))

            s_content = _normalize({k: v for k, v in score_content(c.user_items, fsq_ids, tfidf_mat).items() if k in candidate_set and k not in seen})
            s_item = _normalize({k: v for k, v in score_co_visitation(c.user_items, co_mat, id_to_idx, idx_to_id).items() if k in candidate_set and k not in seen})

            mk = dict(next_poi_order2(c.prev_poi, c.current_poi, trans2, trans1, topn=150, backoff=0.35))
            mk = {k: v for k, v in mk.items() if k in candidate_set and k not in seen}
            mk = _normalize(_apply_hub_penalty(mk, item_counts, markov_hub_alpha))

            s_embed: Dict[str, float] = {}
            if emb_model is not None:
                ctx = _embedding_context(c.user_items, c.current_poi, emb_context_n)
                eb = score_embeddings_context(emb_model, ctx, topn=min(emb_topn, 3000))
                eb = {k: v for k, v in eb.items() if k in candidate_set and k not in seen}
                s_embed = _normalize(_apply_hub_penalty(eb, item_counts, emb_hub_alpha))

            s_als: Dict[str, float] = {}
            if als_model is not None:
                al = score_als(als_model, c.user_id, c.user_items, user_to_idx, item_to_idx, idx_to_item, topn=800)
                al = {k: v for k, v in al.items() if k in candidate_set and k not in seen}
                s_als = _normalize(al)

            comb = _combine(w, s_content, s_item, mk, s_embed, s_als)
            recs = [k for k, _ in sorted(comb.items(), key=lambda x: x[1], reverse=True)[: int(args.k)]]
            mets.append(compute_metrics(recs, c.truth, int(args.k)))

        df = pd.DataFrame(mets)
        return {
            "hit": float(df["hit"].mean()),
            "mrr": float(df["mrr"].mean()),
            "ndcg": float(df["ndcg"].mean()),
            "recall": float(df["recall"].mean()),
        }

    scored: List[Tuple[Tuple[float, float, float, float, float], Dict[str, float]]] = []
    for w in candidates:
        m = _eval_weights(w)
        scored.append((w, m))

    # Pick best by NDCG then MRR.
    scored.sort(key=lambda x: (x[1]["ndcg"], x[1]["mrr"]), reverse=True)
    best_w, best_m = scored[0]

    out = {
        "city_qid": args.city_qid,
        "k": int(args.k),
        "visits_limit": int(args.visits_limit),
        "test_size": int(args.test_size),
        "min_train": int(args.min_train),
        "max_cases": int(args.max_cases),
        "trials": int(args.trials),
        "best_weights": list(best_w),
        "best_metrics": best_m,
        "all_trials": [{"weights": list(w), "metrics": m} for (w, m) in scored],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Best weights (hybrid.trail_current):", list(best_w))
    print("Best metrics:", best_m)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
