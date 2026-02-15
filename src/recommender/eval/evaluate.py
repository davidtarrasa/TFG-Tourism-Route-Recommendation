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
    Split por trail: para cada trail_id, deja las Ãºltimas `test_size` visitas como test.

    Esto alinea mejor la evaluaciÃ³n con modelos secuenciales (Markov/Word2Vec),
    porque el "POI actual" y el siguiente pertenecen a la misma ruta/sesiÃ³n.
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


def compute_metrics(recs: List[str], truth: Iterable[str], k: int) -> Dict[str, float]:
    truth_set = set(truth)
    if not truth_set or not recs:
        return {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    recs_at_k = recs[:k]
    hits = [i for i, r in enumerate(recs_at_k) if r in truth_set]
    hit_flag = 1.0 if hits else 0.0
    recall = len(hits) / len(truth_set)
    mrr = 0.0
    ndcg = 0.0
    if hits:
        rank = min(hits) + 1  # 1-based
        mrr = 1.0 / rank
        ndcg = 1.0 / np.log2(rank + 1)
    return {"hit": hit_flag, "recall": recall, "mrr": mrr, "ndcg": ndcg}


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
) -> pd.DataFrame:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    hyb_cfg = cfg.get("hybrid", {})
    emb_cfg = cfg.get("embeddings", {})
    als_cfg = cfg.get("als", {})
    eval_cfg = cfg.get("eval", {})
    markov_cfg = cfg.get("markov", {})

    # Popularity counts for hub penalty (avoid ultra-popular POIs dominating results).
    item_counts: Dict[str, int] = train_visits["venue_id"].astype(str).value_counts().to_dict()
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
    else:
        cases = list(test_visits["user_id"].unique())

    # Make the evaluated subset reproducible when limiting cases.
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(cases)
    if max_users:
        cases = cases[:max_users]

    metrics_acc = defaultdict(list)
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
                embed_scores = score_embeddings_next(embed_model, str(current_poi), topn=topn)
            embed_scores = _apply_hub_penalty(embed_scores, item_counts=item_counts, alpha=float(emb_cfg.get("hub_alpha", 0.0)))
            # Remove already-seen items early; embeddings often rank near-duplicates from the same trail very high.
            if seen:
                embed_scores = {k: v for k, v in embed_scores.items() if k not in seen}

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
        content_scores = {k: v for k, v in content_scores.items() if k in allowed_ids}
        co_scores = {k: v for k, v in co_scores.items() if k in allowed_ids}
        markov_scores = {k: v for k, v in markov_scores.items() if k in allowed_ids}
        embed_scores = {k: v for k, v in embed_scores.items() if k in allowed_ids}

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
            else:  # hybrid
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

            if not rec_scores:
                continue
            recs_sorted = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
            # Evaluate against *new* POIs -> remove already-seen items from the recommendation list.
            rec_ids = [r[0] for r in recs_sorted if r[0] not in seen][:k]
            m = compute_metrics(rec_ids, truth_items, k)
            for key, val in m.items():
                metrics_acc[(mode, key)].append(val)
        users_evaluated.add(uid)

    # Agregar métricas
    rows = []
    for (mode, metric), vals in metrics_acc.items():
        rows.append({"mode": mode, "metric": metric, "value": float(np.mean(vals)) if vals else 0.0, "n_users": len(vals)})
    summary = {
        "users_evaluated": len(users_evaluated),
        "users_skipped_all_truth_seen": users_skipped_all_truth_seen,
    }
    return pd.DataFrame(rows), summary


def main():
    parser = argparse.ArgumentParser(description="Evaluación offline (hold-out por usuario)")
    parser.add_argument("--city", help="Filtro opcional sobre pois.city")
    parser.add_argument("--city-qid", dest="city_qid", help="Filtro opcional sobre visits.venue_city (QID)")
    parser.add_argument("--k", type=int, default=10, help="Top-K para métricas")
    parser.add_argument("--test-size", type=int, default=1, help="Últimas visitas al test")
    parser.add_argument("--min-train", type=int, default=1, help="Mínimo de visitas en train por usuario")
    parser.add_argument(
        "--protocol",
        choices=["user", "trail"],
        default="user",
        help="Como se hace el split: user (por usuario) o trail (por ruta).",
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
        help="Entrena modelos SOLO con el split de train (mÃ¡s lento, sin fuga de informaciÃ³n).",
    )
    parser.add_argument("--modes", nargs="+", default=["hybrid", "content", "item", "markov", "embed", "als"], help="Modos a evaluar")
    parser.add_argument("--output", help="Guardar resultados en JSON/CSV (según extensión)")
    args = parser.parse_args()
    cfg = load_config(DEFAULT_CONFIG_PATH)
    eval_cfg = cfg.get("eval", {})
    seed = args.seed if args.seed is not None else eval_cfg.get("seed")

    conn = get_conn()
    visits = load_visits(conn, city_qid=args.city_qid, limit=args.visits_limit)
    pois = load_pois(conn, city=args.city, city_qid=args.city_qid)
    poi_cats = load_poi_categories(conn, fsq_ids=pois["fsq_id"])

    if args.protocol == 'trail':
        train_visits, test_visits = split_train_test_trails(visits, test_size=args.test_size, min_train=args.min_train)
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
        print(f"Usuarios saltados (test todo repetido): {summary.get('users_skipped_all_truth_seen', 0)}")
        print(df_metrics.pivot(index="mode", columns="metric", values="value"))


if __name__ == "__main__":
    main()
