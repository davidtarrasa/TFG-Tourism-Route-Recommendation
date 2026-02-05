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
from ..features.transitions import build_transitions
from ..features.word2vec import sequences_from_visits
from ..models.content_based import score_content
from ..models.co_visitation import score_co_visitation
from ..models.markov import next_poi
from ..models.embeddings import score_embeddings, train_embeddings
from ..utils_db import get_conn, load_pois, load_poi_categories, load_visits


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
) -> pd.DataFrame:
    # Features globales sobre train
    tfidf_matrix, tfidf_ids, _ = build_tfidf(poi_cats)
    co_mat, id_to_idx, idx_to_id = build_cooccurrence(train_visits)
    trans_poi, _ = build_transitions(train_visits, pois)

    embed_model = None
    if use_embeddings and embeddings_path:
        try:
            import joblib
            embed_model = joblib.load(embeddings_path)
        except Exception:
            seqs = sequences_from_visits(train_visits)
            embed_model = train_embeddings(seqs)
            joblib.dump(embed_model, embeddings_path)

    als_obj = None
    if use_als and als_path:
        try:
            import joblib
            als_obj = joblib.load(als_path)
        except Exception:
            als_obj = None

    users = list(test_visits["user_id"].unique())
    if max_users:
        users = users[:max_users]

    metrics_acc = defaultdict(list)
    users_evaluated = set()

    for uid in users:
        user_train = train_visits[train_visits["user_id"] == uid]
        user_test = test_visits[test_visits["user_id"] == uid]
        if user_train.empty or user_test.empty:
            continue

        user_items = user_train["venue_id"].astype(str).tolist()
        truth_items = user_test["venue_id"].astype(str).tolist()

        # POI actual para Markov: última del train
        current_poi = user_train.iloc[-1]["venue_id"]

        content_scores = score_content(user_items, tfidf_ids, tfidf_matrix)
        co_scores = score_co_visitation(user_items, co_mat, id_to_idx, idx_to_id)
        markov_scores = dict(next_poi(current_poi, trans_poi, topn=500))
        embed_scores = {}
        if embed_model:
            embed_scores = score_embeddings(embed_model, user_items, topn=1000)

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
                topn=500,
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
                # Alineado con scorer: más peso a item/markov
                if user_items and current_poi:
                    w = (0.3, 0.25, 0.35, 0.05, 0.05 if als_scores else 0.0)
                elif user_items:
                    w = (0.3, 0.35, 0.2, 0.05, 0.1 if als_scores else 0.0)
                elif current_poi:
                    w = (0.2, 0.3, 0.4, 0.05, 0.05 if als_scores else 0.0)
                elif embed_scores or als_scores:
                    w = (0.25, 0.3, 0.2, 0.15 if embed_scores else 0.0, 0.1 if als_scores else 0.0)
                else:
                    w = (0.6, 0.2, 0.2, 0.0, 0.0)
                rec_scores = combine(*w)

            if not rec_scores:
                continue
            recs_sorted = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
            rec_ids = [r[0] for r in recs_sorted[:k]]
            m = compute_metrics(rec_ids, truth_items, k)
            for key, val in m.items():
                metrics_acc[(mode, key)].append(val)
        users_evaluated.add(uid)

    # Agregar métricas
    rows = []
    for (mode, metric), vals in metrics_acc.items():
        rows.append({"mode": mode, "metric": metric, "value": float(np.mean(vals)) if vals else 0.0, "n_users": len(vals)})
    summary = {"users_evaluated": len(users_evaluated)}
    return pd.DataFrame(rows), summary


def main():
    parser = argparse.ArgumentParser(description="Evaluación offline (hold-out por usuario)")
    parser.add_argument("--city", help="Filtro opcional sobre pois.city")
    parser.add_argument("--city-qid", dest="city_qid", help="Filtro opcional sobre visits.venue_city (QID)")
    parser.add_argument("--k", type=int, default=10, help="Top-K para métricas")
    parser.add_argument("--test-size", type=int, default=1, help="Últimas visitas al test")
    parser.add_argument("--min-train", type=int, default=1, help="Mínimo de visitas en train por usuario")
    parser.add_argument("--max-users", type=int, help="Limitar número de usuarios evaluados")
    parser.add_argument("--visits-limit", type=int, help="Limitar visits para acelerar")
    parser.add_argument("--use-embeddings", action="store_true", help="Usar embeddings Word2Vec si existen")
    parser.add_argument("--embeddings-path", help="Ruta del modelo Word2Vec", default="src/recommender/cache/word2vec.joblib")
    parser.add_argument("--use-als", action="store_true", help="Usar modelo ALS si existe")
    parser.add_argument("--als-path", help="Ruta del modelo ALS", default="src/recommender/cache/als_model.joblib")
    parser.add_argument("--modes", nargs="+", default=["hybrid", "content", "item", "markov", "embed", "als"], help="Modos a evaluar")
    parser.add_argument("--output", help="Guardar resultados en JSON/CSV (según extensión)")
    args = parser.parse_args()

    conn = get_conn()
    visits = load_visits(conn, city_qid=args.city_qid, limit=args.visits_limit)
    pois = load_pois(conn, city=args.city)
    poi_cats = load_poi_categories(conn, fsq_ids=pois["fsq_id"])

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
        print(df_metrics.pivot(index="mode", columns="metric", values="value"))


if __name__ == "__main__":
    main()
