"""
Evaluación offline (hold-out por usuario) para los modos del recomendador.

Estrategia:
- Split por usuario: últimas N visitas al test, el resto al train (se descartan usuarios con poco histórico).
- Se generan recomendaciones con las señales del train y se comparan con los ítems de test.
- Métricas: HitRate@k, Recall@k, MRR@k, NDCG@k.

Modos soportados: hybrid, content, item, markov, embed (si hay modelo Word2Vec en disco).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..features.tfidf import build_tfidf
from ..features.cooccurrence import build_cooccurrence
from ..features.transitions import build_transitions, build_transitions_order2
from ..features.word2vec import sequences_from_visits
from ..models.content_based import score_content
from ..models.co_visitation import score_co_visitation
from ..models.markov import next_poi_order2
from ..models.embeddings import score_embeddings_context, score_embeddings_next, train_embeddings
from ..models.als import build_interactions, train_als
from ..utils_db import get_conn, load_pois, load_poi_categories, load_visits
from ..config import DEFAULT_CONFIG_PATH, load_config


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


def _embedding_context(user_items: List[str], current_poi: Optional[str], context_n: int) -> List[str]:
    ctx = [str(x) for x in user_items if x]
    if current_poi:
        ctx = [x for x in ctx if x != str(current_poi)]
        ctx.append(str(current_poi))
    if not ctx:
        return []
    if context_n <= 0:
        return ctx[-1:]
    return ctx[-context_n:]


def split_train_test(visits: pd.DataFrame, test_size: int = 1, min_train: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split por usuario: últimas `test_size` visitas a test, resto a train."""
    visits = visits.sort_values("timestamp")
    train_rows = []
    test_rows = []
    for uid, group in visits.groupby("user_id"):
        if len(group) <= test_size + min_train - 1:
            continue
        test_part = group.tail(test_size)
        train_part = group.iloc[:-test_size]
        if len(train_part) < min_train:
            continue
        train_rows.append(train_part)
        test_rows.append(test_part)
    if not train_rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(train_rows), pd.concat(test_rows)


def split_train_test_trails(visits: pd.DataFrame, test_size: int = 1, min_train: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split por trail: para cada trail_id, deja las últimas `test_size` visitas como test.

    Esto alinea mejor la evaluación con modelos secuenciales (Markov/Word2Vec),
    porque el "POI actual" y el siguiente pertenecen a la misma ruta/sesión.
    """
    visits = visits.sort_values(["trail_id", "timestamp"])
    train_rows = []
    test_rows = []
    for tid, group in visits.groupby("trail_id"):
        if len(group) <= test_size + min_train - 1:
            continue
        test_part = group.tail(test_size)
        train_part = group.iloc[:-test_size]
        if len(train_part) < min_train:
            continue
        train_rows.append(train_part)
        test_rows.append(test_part)
    if not train_rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(train_rows), pd.concat(test_rows)


def split_train_test_last_trail_user(
    visits: pd.DataFrame,
    min_train: int = 1,
    min_test_pois: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split por usuario usando la última ruta completa como test.

    Para cada usuario:
    - test: todas las visitas de su último trail (por timestamp final del trail)
    - train: el resto de visitas del usuario
    """
    if visits.empty:
        return pd.DataFrame(), pd.DataFrame()

    visits = visits.sort_values("timestamp")
    train_rows = []
    test_rows = []

    for uid, group in visits.groupby("user_id"):
        if group.empty or group["trail_id"].isna().all():
            train_rows.append(group)
            continue

        trail_ends = group.groupby("trail_id")["timestamp"].max().sort_values()
        # Keep users with only one trail entirely in TRAIN.
        if trail_ends.empty or len(trail_ends) < 2:
            train_rows.append(group)
            continue

        last_trail_id = trail_ends.index[-1]
        test_part = group[group["trail_id"] == last_trail_id]
        train_part = group[group["trail_id"] != last_trail_id]

        # Test only if last trail is meaningful (>= min_test_pois), otherwise keep all in TRAIN.
        if len(test_part) >= int(min_test_pois) and len(train_part) >= min_train:
            train_rows.append(train_part)
            test_rows.append(test_part)
        else:
            train_rows.append(group)

    train_df = pd.concat(train_rows) if train_rows else pd.DataFrame()
    test_df = pd.concat(test_rows) if test_rows else pd.DataFrame()
    return train_df, test_df


def _novelty_at_k(rec_ids: List[str], item_counts: Dict[str, int]) -> float:
    """Average novelty in [0,1], higher means less popular recommendations."""
    if not rec_ids:
        return 0.0
    total = float(sum(item_counts.values()))
    vocab = float(max(len(item_counts), 1))
    denom = total + vocab
    max_novelty = -np.log2(1.0 / denom) if denom > 1 else 1.0
    vals = []
    for rid in rec_ids:
        c = float(item_counts.get(str(rid), 0))
        p = (c + 1.0) / denom
        vals.append(float(-np.log2(p)) / max_novelty if max_novelty > 0 else 0.0)
    return float(np.mean(vals))


def _diversity_at_k(rec_ids: List[str], poi_primary_map: Dict[str, str]) -> float:
    """Category coverage ratio in [0,1]."""
    if not rec_ids:
        return 0.0
    cats = []
    for rid in rec_ids:
        c = poi_primary_map.get(str(rid))
        cats.append(str(c) if c and str(c) != "nan" else "__unknown__")
    return float(len(set(cats)) / len(cats))


def _ndcg_from_binary_relevance(relevance: List[float], ideal_ones: int) -> float:
    if not relevance:
        return 0.0
    dcg = 0.0
    for i, rel in enumerate(relevance):
        if rel > 0:
            dcg += float(rel) / np.log2(i + 2.0)
    idcg = 0.0
    for i in range(int(max(ideal_ones, 0))):
        idcg += 1.0 / np.log2(i + 2.0)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _normalize_category(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _stable_case_seed(base_seed: int, protocol: str, case_id: object) -> int:
    payload = f"{protocol}:{case_id}".encode("utf-8", errors="ignore")
    h = hashlib.blake2b(payload, digest_size=8).digest()
    v = int.from_bytes(h, byteorder="little", signed=False)
    return int((int(base_seed) + v) % (2**32 - 1))


def compute_metrics(recs: List[str], truth: Iterable[str], k: int) -> Dict[str, float]:
    truth_set = set(str(x) for x in truth)
    if not truth_set or not recs:
        return {"hit": 0.0, "precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    recs_at_k = [str(x) for x in recs[:k]]
    relevance = [1.0 if r in truth_set else 0.0 for r in recs_at_k]
    hits = [i for i, rel in enumerate(relevance) if rel > 0]
    hit_flag = 1.0 if hits else 0.0
    precision = float(sum(relevance) / max(int(k), 1))
    recall = float(sum(relevance) / max(len(truth_set), 1))
    mrr = 0.0
    if hits:
        mrr = 1.0 / float(min(hits) + 1)
    ndcg = _ndcg_from_binary_relevance(relevance, ideal_ones=min(len(truth_set), int(k)))
    return {"hit": hit_flag, "precision": precision, "recall": recall, "mrr": mrr, "ndcg": ndcg}


def compute_category_metrics(
    recs: List[str],
    truth: Iterable[str],
    k: int,
    poi_primary_map: Dict[str, str],
) -> Dict[str, float]:
    truth_cats = {
        c
        for poi in truth
        for c in [_normalize_category(poi_primary_map.get(str(poi)))]
        if c is not None
    }
    if not truth_cats or not recs:
        return {"hit": 0.0, "precision": 0.0, "recall": 0.0, "ndcg": 0.0}

    rec_cats = [
        _normalize_category(poi_primary_map.get(str(poi)))
        for poi in recs[:k]
    ]

    # Category-level relevance is counted once per relevant category.
    # Repeating the same category should not keep increasing gain.
    seen_relevant_cats = set()
    relevance: List[float] = []
    for c in rec_cats:
        if c is None or c not in truth_cats:
            relevance.append(0.0)
            continue
        if c in seen_relevant_cats:
            relevance.append(0.0)
            continue
        seen_relevant_cats.add(c)
        relevance.append(1.0)

    hit_flag = 1.0 if any(rel > 0 for rel in relevance) else 0.0
    precision = float(sum(relevance) / max(int(k), 1))
    matched_cats = {c for c in rec_cats if c is not None and c in truth_cats}
    recall = float(len(matched_cats) / max(len(truth_cats), 1))
    ndcg = _ndcg_from_binary_relevance(relevance, ideal_ones=min(len(truth_cats), int(k)))
    return {"hit": hit_flag, "precision": precision, "recall": recall, "ndcg": ndcg}


def eval_modes(
    train_visits: pd.DataFrame,
    test_visits: pd.DataFrame,
    poi_cats: pd.DataFrame,
    pois: pd.DataFrame,
    modes: List[str],
    k: int,
    max_users: Optional[int] = None,
    use_embeddings: bool = False,
    embeddings_path: Optional[str] = None,
    use_als: bool = False,
    als_path: Optional[str] = None,
    fair: bool = False,
    protocol: str = "user",
    seed: Optional[int] = None,
    city_qid: Optional[str] = None,
) -> pd.DataFrame:
    cfg = load_config(DEFAULT_CONFIG_PATH, city_qid=city_qid)
    hyb_cfg = cfg.get("hybrid", {})
    emb_cfg = cfg.get("embeddings", {})
    als_cfg = cfg.get("als", {})
    eval_cfg = cfg.get("eval", {})
    markov_cfg = cfg.get("markov", {})

    # Popularity counts for hub penalty (avoid ultra-popular POIs dominating results).
    item_counts: Dict[str, int] = train_visits["venue_id"].astype(str).value_counts().to_dict()
    poi_primary_map: Dict[str, str] = {}
    if not pois.empty and "fsq_id" in pois.columns and "primary_category" in pois.columns:
        poi_primary_map = {
            str(k): str(v)
            for k, v in pois.set_index("fsq_id")["primary_category"].to_dict().items()
            if _normalize_category(v) is not None
        }
    if not poi_cats.empty and {"fsq_id", "category_name"}.issubset(set(poi_cats.columns)):
        fallback_cats = (
            poi_cats.dropna(subset=["category_name"])
            .groupby("fsq_id", as_index=False)["category_name"]
            .first()
            .set_index("fsq_id")["category_name"]
            .to_dict()
        )
        for pid, cat in fallback_cats.items():
            key = str(pid)
            if key not in poi_primary_map and _normalize_category(cat) is not None:
                poi_primary_map[key] = str(cat)
    # Features globales sobre train
    tfidf_matrix, tfidf_ids, _ = build_tfidf(poi_cats)
    co_mat, id_to_idx, idx_to_id = build_cooccurrence(train_visits)
    trans_poi, _ = build_transitions(train_visits, pois)
    trans_poi2 = build_transitions_order2(train_visits)

    # IMPORTANT:
    # - fair=False (default): load pre-trained models from disk (fast, but can be optimistic).
    # - fair=True: train models ONLY on train_visits (slower, but leak-free).
    embed_model = None
    if use_embeddings:
        if (not fair) and embeddings_path:
            try:
                import joblib
                embed_model = joblib.load(embeddings_path)
            except Exception:
                embed_model = None

        if embed_model is None:
            seqs = sequences_from_visits(train_visits)
            # Use lighter params under fair evaluation to keep runtime reasonable.
            e_vector_size = int(eval_cfg.get("emb_vector_size", emb_cfg.get("vector_size", 128))) if fair else int(emb_cfg.get("vector_size", 128))
            e_window = int(eval_cfg.get("emb_window", emb_cfg.get("window", 15))) if fair else int(emb_cfg.get("window", 15))
            e_epochs = int(eval_cfg.get("emb_epochs", emb_cfg.get("epochs", 10))) if fair else int(emb_cfg.get("epochs", 10))
            e_negative = int(eval_cfg.get("emb_negative", emb_cfg.get("negative", 5))) if fair else int(emb_cfg.get("negative", 5))
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
            # Do not overwrite the "real" cached model during fair evaluation runs.
            if (not fair) and embeddings_path:
                try:
                    import joblib
                    joblib.dump(embed_model, embeddings_path)
                except Exception:
                    pass

    als_obj = None
    if use_als:
        if (not fair) and als_path:
            try:
                import joblib
                als_obj = joblib.load(als_path)
            except Exception:
                als_obj = None

        if als_obj is None:
            interactions, user_to_idx, item_to_idx, idx_to_item = build_interactions(train_visits)
            # Use lighter params under fair evaluation to keep runtime reasonable.
            a_factors = int(eval_cfg.get("als_factors", als_cfg.get("factors", 64))) if fair else int(als_cfg.get("factors", 64))
            a_reg = float(eval_cfg.get("als_regularization", als_cfg.get("regularization", 0.01))) if fair else float(als_cfg.get("regularization", 0.01))
            a_iters = int(eval_cfg.get("als_iterations", als_cfg.get("iterations", 15))) if fair else int(als_cfg.get("iterations", 15))
            a_alpha = float(eval_cfg.get("als_alpha", als_cfg.get("alpha", 40.0))) if fair else float(als_cfg.get("alpha", 40.0))
            als_model = train_als(
                interactions,
                factors=a_factors,
                regularization=a_reg,
                iterations=a_iters,
                alpha=a_alpha,
            )
            als_obj = {
                "model": als_model,
                "user_to_idx": user_to_idx,
                "item_to_idx": item_to_idx,
                "idx_to_item": idx_to_item,
            }
            if als_path:
                try:
                    import joblib
                    # Do not overwrite the "real" cached model during fair evaluation runs.
                    if not fair:
                        joblib.dump(als_obj, als_path)
                except Exception:
                    pass

    if protocol == "trail":
        cases = list(test_visits["trail_id"].unique())
    elif protocol == "last_trail_user":
        cases = list(test_visits["user_id"].unique())
    else:
        cases = list(test_visits["user_id"].unique())

    # Make the evaluated subset reproducible when limiting cases.
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(cases)
    if max_users:
        cases = cases[:max_users]

    metrics_acc = defaultdict(list)
    group_metrics_acc: Dict = defaultdict(list)
    cold_warm_users: Dict[str, set] = {"cold": set(), "warm": set()}
    users_evaluated = set()
    users_skipped_all_truth_seen = 0

    for case_id in cases:
        # Case context:
        # - protocol=user  -> evaluate next POI for a user
        # - protocol=trail -> evaluate next POI inside the same trail/session
        if protocol == "trail":
            trail_train = train_visits[train_visits["trail_id"] == case_id].sort_values("timestamp")
            trail_test = test_visits[test_visits["trail_id"] == case_id].sort_values("timestamp")
            if trail_train.empty or trail_test.empty:
                continue
            uid = int(trail_train.iloc[-1]["user_id"])

            # Profile/history-based models (content/item/ALS) can use user history.
            user_train = train_visits[train_visits["user_id"] == uid].sort_values("timestamp")
            user_items = user_train["venue_id"].astype(str).tolist()

            # Sequential models (markov/embed) should be anchored in the trail.
            seq_items = trail_train["venue_id"].astype(str).tolist()
            seen = set(seq_items)
            truth_items = [t for t in trail_test["venue_id"].astype(str).tolist() if t not in seen]
            if not truth_items:
                users_skipped_all_truth_seen += 1
                continue

            current_poi = str(seq_items[-1])
            prev_poi = str(seq_items[-2]) if len(seq_items) >= 2 else None
        elif protocol == "last_trail_user":
            uid = int(case_id)
            user_train = train_visits[train_visits["user_id"] == uid].sort_values("timestamp")
            user_test = test_visits[test_visits["user_id"] == uid].sort_values("timestamp")
            if user_train.empty or user_test.empty:
                continue

            user_items = user_train["venue_id"].astype(str).tolist()
            test_seq = user_test["venue_id"].astype(str).tolist()
            if len(test_seq) < 2:
                continue

            # Seed = first POI of the held-out last trail, truth = the remaining POIs.
            current_poi = str(test_seq[0])
            prev_poi = None
            seen = set(user_items)
            seen.add(current_poi)
            seq_items = user_items + [current_poi]
            truth_items = [t for t in test_seq[1:] if t not in seen]
            if not truth_items:
                users_skipped_all_truth_seen += 1
                continue
        else:
            uid = int(case_id)
            user_train = train_visits[train_visits["user_id"] == uid].sort_values("timestamp")
            user_test = test_visits[test_visits["user_id"] == uid].sort_values("timestamp")
            if user_train.empty or user_test.empty:
                continue

            user_items = user_train["venue_id"].astype(str).tolist()
            seq_items = user_items
            seen = set(user_items)
            truth_items = [t for t in user_test["venue_id"].astype(str).tolist() if t not in seen]
            if not truth_items:
                users_skipped_all_truth_seen += 1
                continue

            current_poi = str(user_train.iloc[-1]["venue_id"])
            prev_poi = str(user_train.iloc[-2]["venue_id"]) if len(user_train) >= 2 else None

        user_group = "cold" if len(user_items) < 5 else "warm"
        content_scores = score_content(user_items, tfidf_ids, tfidf_matrix)
        co_scores = score_co_visitation(user_items, co_mat, id_to_idx, idx_to_id)
        markov_scores = dict(next_poi_order2(prev_poi, str(current_poi), trans_poi2, trans_poi, topn=2000, backoff=0.3))
        markov_scores = _apply_hub_penalty(markov_scores, item_counts=item_counts, alpha=float(markov_cfg.get("hub_alpha", 0.0)))

        embed_scores: Dict[str, float] = {}
        if embed_model:
            topn = int(emb_cfg.get("topn_score", 1000))
            ctx_n = int(emb_cfg.get("context_n", 1))
            ctx = _embedding_context(user_items=seq_items, current_poi=str(current_poi), context_n=ctx_n)
            if len(ctx) >= 2:
                embed_scores = score_embeddings_context(embed_model, ctx, topn=topn)
            else:
                embed_scores = score_embeddings_next(embed_model, str(current_poi), topn=topn, user_items=seq_items)
            embed_scores = _apply_hub_penalty(embed_scores, item_counts=item_counts, alpha=float(emb_cfg.get("hub_alpha", 0.0)))
        als_scores = {}
        if als_obj:
            from ..models.als import score_als

            als_scores = score_als(
                als_obj["model"],
                uid,
                user_items,
                als_obj["user_to_idx"],
                als_obj["item_to_idx"],
                als_obj["idx_to_item"],
                topn=int(als_cfg.get("topn_score", 500)),
            )

        # Normalizar
        def norm(d):
            if not d:
                return {}
            vals = np.array(list(d.values()), dtype=np.float32)
            vmax = vals.max()
            if vmax <= 0:
                return {}
            return {k: v / vmax for k, v in d.items()}

        content_scores = norm(content_scores)
        co_scores = norm(co_scores)
        markov_scores = norm(markov_scores)
        embed_scores = norm(embed_scores)
        als_scores = norm(als_scores)

        allowed_ids = set(pois["fsq_id"].astype(str))
        candidate_pool = {fid for fid in allowed_ids if fid not in seen}

        def filter_scores(scores: Dict[str, float]) -> Dict[str, float]:
            if not scores:
                return {}
            return {pid: val for pid, val in scores.items() if pid in candidate_pool}

        content_scores = filter_scores(content_scores)
        co_scores = filter_scores(co_scores)
        markov_scores = filter_scores(markov_scores)
        embed_scores = filter_scores(embed_scores)
        als_scores = filter_scores(als_scores)

        def combine(w_content, w_co, w_markov, w_embed, w_als):
            all_ids = (
                set(content_scores)
                | set(co_scores)
                | set(markov_scores)
                | set(embed_scores)
                | set(als_scores)
            )
            return {
                fid: w_content * content_scores.get(fid, 0.0)
                + w_co * co_scores.get(fid, 0.0)
                + w_markov * markov_scores.get(fid, 0.0)
                + w_embed * embed_scores.get(fid, 0.0)
                + w_als * als_scores.get(fid, 0.0)
                for fid in all_ids
            }

        case_seed = int(seed if seed is not None else eval_cfg.get("seed", 42))
        case_rng = np.random.default_rng(_stable_case_seed(case_seed, protocol, case_id))

        for mode in modes:
            if mode == "content":
                rec_scores = content_scores
            elif mode == "item":
                rec_scores = co_scores
            elif mode == "markov":
                rec_scores = markov_scores
            elif mode == "embed":
                rec_scores = embed_scores
            elif mode == "als":
                rec_scores = als_scores
            elif mode == "hybrid":
                if protocol == "trail" and current_poi:
                    w = list(hyb_cfg.get("trail_current", [0.05, 0.10, 0.55, 0.30, 0.00]))
                elif user_items and current_poi:
                    w = list(hyb_cfg.get("user_current", [0.1, 0.15, 0.15, 0.05, 0.55]))
                elif user_items:
                    w = list(hyb_cfg.get("user_only", [0.1, 0.2, 0.1, 0.05, 0.55]))
                elif current_poi:
                    w = list(hyb_cfg.get("current_only", [0.1, 0.15, 0.35, 0.05, 0.35]))
                elif embed_scores or als_scores:
                    w = list(hyb_cfg.get("embed_or_als", [0.15, 0.15, 0.1, 0.15, 0.45]))
                else:
                    w = list(hyb_cfg.get("cold_start", [0.6, 0.2, 0.2, 0.0, 0.0]))

                # If ALS/embeddings are not available, zero their weights.
                if not als_scores:
                    w[4] = 0.0
                if not embed_scores:
                    w[3] = 0.0
                rec_scores = combine(*w)
            elif mode == "random":
                if not candidate_pool:
                    continue
                sample = list(candidate_pool)
                sample.sort()
                case_rng.shuffle(sample)
                rec_ids = sample[:k]
                m = compute_metrics(rec_ids, truth_items, k)
                cm = compute_category_metrics(rec_ids, truth_items, k, poi_primary_map=poi_primary_map)
                novelty = _novelty_at_k(rec_ids, item_counts=item_counts)
                diversity = _diversity_at_k(rec_ids, poi_primary_map=poi_primary_map)
                for _mk, _mv in [("hit", m.get("hit", 0.0)), ("precision", m.get("precision", 0.0)),
                                  ("recall", m.get("recall", 0.0)), ("ndcg", m.get("ndcg", 0.0)),
                                  ("cat_hit", cm.get("hit", 0.0)), ("cat_precision", cm.get("precision", 0.0)),
                                  ("cat_recall", cm.get("recall", 0.0)), ("cat_ndcg", cm.get("ndcg", 0.0)),
                                  ("novelty", novelty), ("diversity", diversity)]:
                    metrics_acc[(mode, _mk)].append(float(_mv))
                    group_metrics_acc[(mode, _mk, user_group)].append(float(_mv))
                continue
            elif mode == "popular":
                # Baseline de popularidad global: recomienda los POIs más visitados en train.
                if not candidate_pool:
                    continue
                rec_ids = sorted(candidate_pool, key=lambda x: item_counts.get(x, 0), reverse=True)[:k]
                m = compute_metrics(rec_ids, truth_items, k)
                cm = compute_category_metrics(rec_ids, truth_items, k, poi_primary_map=poi_primary_map)
                novelty = _novelty_at_k(rec_ids, item_counts=item_counts)
                diversity = _diversity_at_k(rec_ids, poi_primary_map=poi_primary_map)
                for _mk, _mv in [("hit", m.get("hit", 0.0)), ("precision", m.get("precision", 0.0)),
                                  ("recall", m.get("recall", 0.0)), ("ndcg", m.get("ndcg", 0.0)),
                                  ("cat_hit", cm.get("hit", 0.0)), ("cat_precision", cm.get("precision", 0.0)),
                                  ("cat_recall", cm.get("recall", 0.0)), ("cat_ndcg", cm.get("ndcg", 0.0)),
                                  ("novelty", novelty), ("diversity", diversity)]:
                    metrics_acc[(mode, _mk)].append(float(_mv))
                    group_metrics_acc[(mode, _mk, user_group)].append(float(_mv))
                continue
            elif mode == "rrf":
                # Reciprocal Rank Fusion: combina los rankings de todos los modelos
                # sin necesitar pesos manuales. score = Σ 1/(k+rank_i), k=60.
                def _ranks(scores: Dict[str, float]) -> Dict[str, int]:
                    return {fid: i for i, fid in enumerate(sorted(scores, key=scores.__getitem__, reverse=True))}
                all_model_ranks = [_ranks(s) for s in [content_scores, co_scores, markov_scores, embed_scores, als_scores]]
                rrf_k = 60
                all_rrf_ids = set(content_scores) | set(co_scores) | set(markov_scores) | set(embed_scores) | set(als_scores)
                rec_scores = {
                    fid: sum(1.0 / (rrf_k + r.get(fid, len(r)) + 1) for r in all_model_ranks)
                    for fid in all_rrf_ids
                }
            else:
                continue

            if not rec_scores:
                continue
            recs_sorted = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
            rec_ids = [r[0] for r in recs_sorted][:k]
            m = compute_metrics(rec_ids, truth_items, k)
            cm = compute_category_metrics(rec_ids, truth_items, k, poi_primary_map=poi_primary_map)
            novelty = _novelty_at_k(rec_ids, item_counts=item_counts)
            diversity = _diversity_at_k(rec_ids, poi_primary_map=poi_primary_map)

            # Main metrics requested by tutor (+ hit for quick compatibility checks).
            for _mk, _mv in [("hit", m.get("hit", 0.0)), ("precision", m.get("precision", 0.0)),
                              ("recall", m.get("recall", 0.0)), ("ndcg", m.get("ndcg", 0.0)),
                              ("cat_hit", cm.get("hit", 0.0)), ("cat_precision", cm.get("precision", 0.0)),
                              ("cat_recall", cm.get("recall", 0.0)), ("cat_ndcg", cm.get("ndcg", 0.0)),
                              ("novelty", novelty), ("diversity", diversity)]:
                metrics_acc[(mode, _mk)].append(float(_mv))
                group_metrics_acc[(mode, _mk, user_group)].append(float(_mv))
        users_evaluated.add(uid)
        cold_warm_users[user_group].add(uid)

    # Agregar métricas
    rows = []
    for (mode, metric), vals in metrics_acc.items():
        rows.append({"mode": mode, "metric": metric, "value": float(np.mean(vals)) if vals else 0.0, "n_users": len(vals)})

    cold_warm_breakdown: Dict = {}
    for (mode, metric, group), vals in group_metrics_acc.items():
        cold_warm_breakdown.setdefault(mode, {}).setdefault(metric, {})[group] = (
            float(np.mean(vals)) if vals else 0.0
        )

    summary = {
        "users_evaluated": len(users_evaluated),
        "users_skipped_all_truth_seen": users_skipped_all_truth_seen,
        "cold_users": len(cold_warm_users.get("cold", set())),
        "warm_users": len(cold_warm_users.get("warm", set())),
        "cold_warm_breakdown": cold_warm_breakdown,
    }
    return pd.DataFrame(rows), summary


def main():
    parser = argparse.ArgumentParser(description="Evaluación offline (hold-out por usuario)")
    parser.add_argument("--city", help="Filtro opcional sobre pois.city")
    parser.add_argument("--city-qid", dest="city_qid", help="Filtro opcional sobre visits.venue_city (QID)")
    parser.add_argument("--k", type=int, default=10, help="Top-K para métricas")
    parser.add_argument("--test-size", type=int, default=1, help="Últimas visitas al test")
    parser.add_argument("--min-train", type=int, default=1, help="Mínimo de visitas en train por usuario")
    parser.add_argument("--min-test-pois", type=int, default=4, help="Mínimo de POIs en test para last_trail_user")
    parser.add_argument(
        "--protocol",
        choices=["user", "trail", "last_trail_user"],
        default="last_trail_user",
        help="Como se hace el split: user (por usuario), trail (por ruta), last_trail_user (ultima ruta por usuario).",
    )
    parser.add_argument("--max-users", type=int, help="Limitar número de usuarios evaluados")
    parser.add_argument("--visits-limit", type=int, help="Limitar visits para acelerar")
    parser.add_argument("--seed", type=int, help="Semilla para muestreo reproducible (cuando hay límites)")
    parser.add_argument("--use-embeddings", action="store_true", help="Usar embeddings Word2Vec si existen")
    parser.add_argument("--embeddings-path", help="Ruta del modelo Word2Vec", default="src/recommender/cache/word2vec.joblib")
    parser.add_argument("--use-als", action="store_true", help="Usar modelo ALS si existe")
    parser.add_argument("--als-path", help="Ruta del modelo ALS", default="src/recommender/cache/als_model.joblib")
    parser.add_argument(
        "--fair",
        action="store_true",
        help="Entrena modelos SOLO con el split de train (más lento, sin fuga de información).",
    )
    parser.add_argument("--modes", nargs="+", default=["hybrid", "content", "item", "markov", "embed", "als", "random", "popular", "rrf"], help="Modos a evaluar")
    parser.add_argument("--output", help="Guardar resultados en JSON/CSV (según extensión)")
    args = parser.parse_args()
    cfg = load_config(DEFAULT_CONFIG_PATH, city_qid=args.city_qid)
    eval_cfg = cfg.get("eval", {})
    seed = args.seed if args.seed is not None else eval_cfg.get("seed")

    conn = get_conn()
    visits = load_visits(conn, city_qid=args.city_qid, limit=args.visits_limit)
    pois = load_pois(conn, city=args.city, city_qid=args.city_qid)
    poi_cats = load_poi_categories(conn, fsq_ids=pois["fsq_id"])

    if args.protocol == "trail":
        train_visits, test_visits = split_train_test_trails(visits, test_size=args.test_size, min_train=args.min_train)
    elif args.protocol == "last_trail_user":
        train_visits, test_visits = split_train_test_last_trail_user(
            visits,
            min_train=args.min_train,
            min_test_pois=args.min_test_pois,
        )
    else:
        train_visits, test_visits = split_train_test(visits, test_size=args.test_size, min_train=args.min_train)
    if train_visits.empty or test_visits.empty:
        raise SystemExit("Sin datos suficientes para evaluar (ajusta test-size/min-train).")

    df_metrics, summary = eval_modes(
        train_visits=train_visits,
        test_visits=test_visits,
        poi_cats=poi_cats,
        pois=pois,
        modes=args.modes,
        k=args.k,
        max_users=args.max_users,
        use_embeddings=args.use_embeddings,
        embeddings_path=args.embeddings_path,
        use_als=args.use_als,
        als_path=args.als_path,
        fair=args.fair,
        protocol=args.protocol,
        seed=seed,
        city_qid=args.city_qid,
    )

    if args.output:
        if args.output.lower().endswith(".json"):
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(
                    {"summary": summary, "metrics": df_metrics.to_dict(orient="records")},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        else:
            df_metrics.to_csv(args.output, index=False)

    if df_metrics.empty:
        print("Sin métricas (¿sin usuarios válidos?).")
    else:
        print(f"Usuarios evaluados: {summary.get('users_evaluated', 0)}")
        print(f"  cold (<5 visitas train): {summary.get('cold_users', 0)}")
        print(f"  warm (>=5 visitas train): {summary.get('warm_users', 0)}")
        print(f"Usuarios saltados (test todo repetido): {summary.get('users_skipped_all_truth_seen', 0)}")
        print(df_metrics.pivot(index="mode", columns="metric", values="value"))


if __name__ == "__main__":
    main()
