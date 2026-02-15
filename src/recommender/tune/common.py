"""Common utilities for tuning scripts.

We keep tuning scripts small and consistent:
- Load data (city_qid) from Postgres.
- Split train/test with the trail protocol.
- Build a list of evaluation cases (prev,current -> next).
- Evaluate a scoring function into standard next-POI metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..eval.evaluate import compute_metrics, split_train_test_trails


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vmax = max(scores.values())
    if vmax <= 0:
        return {}
    return {k: float(v) / float(vmax) for k, v in scores.items()}


def apply_hub_penalty(scores: Dict[str, float], item_counts: Dict[str, int], alpha: float) -> Dict[str, float]:
    """score' = score / (count ** alpha). alpha=0 disables."""
    if not scores:
        return {}
    if alpha <= 0:
        return scores
    out: Dict[str, float] = {}
    for k, v in scores.items():
        c = float(item_counts.get(str(k), 1))
        out[str(k)] = float(v) / (c**alpha if c > 0 else 1.0)
    return out


@dataclass
class TrailCase:
    user_id: int
    prev_poi: Optional[str]
    current_poi: str
    user_items: List[str]
    truth: List[str]


def build_trail_cases(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[TrailCase]:
    """Trail protocol cases built from split_train_test_trails()."""
    if train_df.empty or test_df.empty:
        return []

    user_hist = (
        train_df.sort_values("timestamp")
        .groupby("user_id")["venue_id"]
        .apply(lambda s: [str(x) for x in s.tolist()])
    )

    cases: List[TrailCase] = []
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
        cases.append(TrailCase(user_id=uid, prev_poi=prev, current_poi=cur, user_items=list(items), truth=truth))
    return cases


def make_train_test_trails(visits: pd.DataFrame, test_size: int, min_train: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[TrailCase]]:
    train_df, test_df = split_train_test_trails(visits, test_size=test_size, min_train=min_train)
    cases = build_trail_cases(train_df, test_df)
    return train_df, test_df, cases


def eval_cases(
    cases: Sequence[TrailCase],
    k: int,
    scorer_fn,
) -> Dict[str, float]:
    """Evaluate a scorer(case)->Dict[item,score] over cases."""
    mets = []
    for c in cases:
        scores: Dict[str, float] = scorer_fn(c)
        recs = [x for x, _ in sorted(scores.items(), key=lambda t: t[1], reverse=True)[:k]]
        mets.append(compute_metrics(recs, c.truth, k))
    df = pd.DataFrame(mets)
    return {
        "hit": float(df["hit"].mean()),
        "mrr": float(df["mrr"].mean()),
        "ndcg": float(df["ndcg"].mean()),
        "recall": float(df["recall"].mean()),
    }


def embedding_context(user_items: List[str], current_poi: str, context_n: int) -> List[str]:
    ctx = [str(x) for x in user_items if x]
    ctx = [x for x in ctx if x != str(current_poi)] + [str(current_poi)]
    if not ctx:
        return []
    if context_n <= 0:
        return ctx[-1:]
    return ctx[-context_n:]

