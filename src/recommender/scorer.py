"""Scorer/orquestador de recomendación.
- Combina puntajes de motores (contenido, co-visitas, Markov y embeddings opcional).
- Aplica filtros básicos (ciudad por carga) y re-ranking por distancia/precio si se indica.
- Devuelve top-K; la ordenación de ruta se añadirá en fases posteriores.
"""

from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd

from .features.load_data import load_all
from .features.tfidf import build_tfidf
from .features.cooccurrence import build_cooccurrence
from .features.transitions import build_transitions, build_transitions_order2
from .features.word2vec import sequences_from_visits
from .models.embeddings import (
    score_embeddings,
    score_embeddings_context,
    score_embeddings_next,
    train_embeddings,
)
from .models.content_based import score_content
from .models.co_visitation import score_co_visitation
from .models.markov import next_poi, next_poi_order2
from .models.als import score_als
from .config import DEFAULT_CONFIG_PATH, load_config
from .prefs import Prefs, normalize_categories
from .route_planner import plan_route

_CFG = None


def _get_cfg():
    global _CFG
    if _CFG is None:
        _CFG = load_config(DEFAULT_CONFIG_PATH)
    return _CFG


def _cfg_get(path: str, default):
    cur = _get_cfg()
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# Categorias que no queremos en el top (salidas raras de Markov)
EXCLUDE_CATEGORIES = set(_cfg_get("filters.exclude_categories", ["Intersection", "State", "Home (private)"]))


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Escala 0-1 por máximo."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float32)
    vmax = vals.max()
    if vmax <= 0:
        return {}
    return {k: v / vmax for k, v in scores.items()}


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
        # Keep current_poi as the most recent item in the context.
        ctx = [x for x in ctx if x != str(current_poi)]
        ctx.append(str(current_poi))
    if not ctx:
        return []
    if context_n <= 0:
        return ctx[-1:]
    return ctx[-context_n:]


def _combine_scores(
    content_scores: Dict[str, float],
    co_scores: Dict[str, float],
    markov_scores: Dict[str, float],
    embed_scores: Optional[Dict[str, float]] = None,
    als_scores: Optional[Dict[str, float]] = None,
    weights: Tuple[float, float, float, float, float] = (0.3, 0.3, 0.3, 0.0, 0.1),
) -> Dict[str, float]:
    """Suma ponderada de los motores."""
    w_content, w_co, w_markov, w_embed, w_als = weights
    all_ids = (
        set(content_scores)
        | set(co_scores)
        | set(markov_scores)
        | (set(embed_scores) if embed_scores else set())
        | (set(als_scores) if als_scores else set())
    )
    combined: Dict[str, float] = {}
    for fid in all_ids:
        combined[fid] = (
            w_content * content_scores.get(fid, 0.0)
            + w_co * co_scores.get(fid, 0.0)
            + w_markov * markov_scores.get(fid, 0.0)
            + w_embed * (embed_scores.get(fid, 0.0) if embed_scores else 0.0)
            + w_als * (als_scores.get(fid, 0.0) if als_scores else 0.0)
        )
    return combined


def recommend(
    dsn: Optional[str] = None,
    city: Optional[str] = None,
    city_qid: Optional[str] = None,
    user_id: Optional[int] = None,
    current_poi: Optional[str] = None,
    k: int = 10,
    visits_limit: Optional[int] = 50000,
    mode: str = "hybrid",
    use_embeddings: bool = False,
    embeddings_path: Optional[str] = "src/recommender/cache/word2vec.joblib",
    use_als: bool = False,
    als_path: Optional[str] = "src/recommender/cache/als_model.joblib",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    max_price_tier: Optional[int] = None,
    free_only: bool = False,
    prefs: Optional[Prefs] = None,
    distance_weight: float = 0.3,
    diversify: bool = True,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame top-K con columnas:
    fsq_id, name, city, primary_category, rating, price_tier, is_free, score (+ distance_km si aplica).
    """
    cfg = load_config(DEFAULT_CONFIG_PATH)
    hyb_cfg = cfg.get("hybrid", {})
    emb_cfg = cfg.get("embeddings", {})
    als_cfg = cfg.get("als", {})
    markov_cfg = cfg.get("markov", {})
    route_pl_cfg = cfg.get("route_planner", {})
    prefs_cfg = cfg.get("prefs", {})

    # 1) Cargar datos (filtra por ciudad si se pasa)
    visits_df, pois_df, poi_cats_df = load_all(dsn=dsn, city=city, city_qid=city_qid, visits_limit=visits_limit)

    # Historial del usuario (si existe)
    user_items: List[str] = []
    if user_id is not None:
        user_items = visits_df[visits_df["user_id"] == user_id]["venue_id"].astype(str).tolist()

    # Popularity counts for hub penalty (avoid ultra-popular POIs dominating results).
    item_counts: Dict[str, int] = visits_df["venue_id"].astype(str).value_counts().to_dict()

    # 2) Features
    tfidf_matrix, tfidf_ids, _ = build_tfidf(poi_cats_df)
    co_mat, id_to_idx, idx_to_id = build_cooccurrence(visits_df)
    trans_poi, trans_cat = build_transitions(visits_df, pois_df)
    trans_poi2 = build_transitions_order2(visits_df)

    # Embeddings opcionales
    embed_scores: Dict[str, float] = {}
    if use_embeddings and mode in ("hybrid", "embed"):
        try:
            import joblib  # type: ignore

            model = None
            if embeddings_path and os.path.exists(embeddings_path):
                model = joblib.load(embeddings_path)
            else:
                seqs = sequences_from_visits(visits_df)
                model = train_embeddings(
                    seqs,
                    vector_size=int(emb_cfg.get("vector_size", 128)),
                    window=int(emb_cfg.get("window", 15)),
                    min_count=int(emb_cfg.get("min_count", 2)),
                    workers=int(emb_cfg.get("workers", 4)),
                    epochs=int(emb_cfg.get("epochs", 10)),
                    negative=int(emb_cfg.get("negative", 5)),
                    sample=float(emb_cfg.get("sample", 1e-3)),
                    ns_exponent=float(emb_cfg.get("ns_exponent", 0.75)),
                    hs=int(emb_cfg.get("hs", 0)),
                    seed=int(emb_cfg.get("seed", 42)),
                )
                if embeddings_path:
                    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
                    joblib.dump(model, embeddings_path)
            if model:
                topn = int(emb_cfg.get("topn_score", 1000))
                ctx_n = int(emb_cfg.get("context_n", 1))
                ctx = _embedding_context(user_items=user_items, current_poi=current_poi, context_n=ctx_n)
                if len(ctx) >= 2:
                    emb_raw = score_embeddings_context(model, ctx, topn=topn)
                elif current_poi:
                    emb_raw = score_embeddings_next(model, str(current_poi), topn=topn)
                else:
                    emb_raw = score_embeddings(model, user_items, topn=topn)
                emb_raw = _apply_hub_penalty(emb_raw, item_counts=item_counts, alpha=float(emb_cfg.get("hub_alpha", 0.0)))
                embed_scores = _normalize_scores(emb_raw)
        except ImportError:
            pass
        except ValueError:
            pass

    als_scores: Dict[str, float] = {}
    if use_als and mode in ("hybrid", "als"):
        try:
            import joblib  # type: ignore

            if als_path and os.path.exists(als_path):
                als_obj = joblib.load(als_path)
                model = als_obj["model"]
                user_to_idx = als_obj["user_to_idx"]
                item_to_idx = als_obj["item_to_idx"]
                idx_to_item = als_obj["idx_to_item"]
                als_scores = score_als(
                    model,
                    user_id or -1,
                    user_items,
                    user_to_idx,
                    item_to_idx,
                    idx_to_item,
                    topn=int(als_cfg.get("topn_score", 500)),
                )
                als_scores = _normalize_scores(als_scores)
        except ImportError:
            pass
        except Exception:
            pass

    allowed_ids = set(pois_df["fsq_id"].astype(str))

    # 3) Scores por motor
    content_scores = score_content(user_items, tfidf_ids, tfidf_matrix) if mode in ("hybrid", "content", "embed") else {}
    co_scores = score_co_visitation(user_items, co_mat, id_to_idx, idx_to_id) if mode in ("hybrid", "item", "embed") else {}
    markov_scores = {}
    if mode in ("hybrid", "markov", "embed") and current_poi:
        prev_poi = None
        if user_items:
            # Best-effort previous POI from history.
            if len(user_items) >= 2 and str(user_items[-1]) == str(current_poi):
                prev_poi = str(user_items[-2])
            else:
                prev_poi = str(user_items[-1])
        markov_scores = dict(next_poi_order2(prev_poi, str(current_poi), trans_poi2, trans_poi, topn=2000, backoff=0.3))
        # Si no hay transiciones POI→POI, usar transiciones de categoría del POI actual
        if not markov_scores:
            current_row = pois_df[pois_df["fsq_id"] == current_poi]
            if not current_row.empty:
                cur_cat = current_row.iloc[0].get("primary_category")
                if cur_cat in trans_cat:
                    cat_next = trans_cat[cur_cat]
                    # Normaliza probs de categorías destino
                    total_cat = sum(cat_next.values()) or 1.0
                    cat_probs = {c: v / total_cat for c, v in cat_next.items()}
                    candidates_cat = pois_df[pois_df["primary_category"].isin(cat_probs.keys())]
                    if not candidates_cat.empty:
                        candidates_cat = candidates_cat.copy()
                        candidates_cat["rating"] = pd.to_numeric(candidates_cat["rating"], errors="coerce").fillna(0)
                        rmax = candidates_cat["rating"].max() or 1.0
                        candidates_cat["r_norm"] = candidates_cat["rating"] / rmax
                        scores = {}
                        for _, row in candidates_cat.iterrows():
                            dest_cat = row["primary_category"]
                            scores[row["fsq_id"]] = cat_probs.get(dest_cat, 0) * (0.7 + 0.3 * row["r_norm"])
                        markov_scores = scores

        # Hub penalty: discourage recommending the same ultra-popular POIs everywhere.
        markov_scores = _apply_hub_penalty(markov_scores, item_counts=item_counts, alpha=float(markov_cfg.get("hub_alpha", 0.0)))

    # Normalizar y limitar a POIs con metadatos
    content_scores = {k: v for k, v in _normalize_scores(content_scores).items() if k in allowed_ids}
    co_scores = {k: v for k, v in _normalize_scores(co_scores).items() if k in allowed_ids}
    markov_scores = {k: v for k, v in _normalize_scores(markov_scores).items() if k in allowed_ids}
    embed_scores = {k: v for k, v in _normalize_scores(embed_scores).items() if k in allowed_ids}
    als_scores = {k: v for k, v in _normalize_scores(als_scores).items() if k in allowed_ids}

    # 4) Combinar según modo
    if mode == "content":
        combined = content_scores
    elif mode == "item":
        combined = co_scores
    elif mode == "markov":
        combined = markov_scores
    elif mode == "embed":
        combined = embed_scores
    elif mode == "als":
        combined = als_scores
    else:
        # Pesos según señales disponibles (reforzamos item/markov)
        if user_items and current_poi:
            w = list(hyb_cfg.get("user_current", [0.1, 0.15, 0.15, 0.05, 0.55]))
        elif user_items:
            w = list(hyb_cfg.get("user_only", [0.1, 0.2, 0.1, 0.05, 0.55]))
        elif current_poi:
            w = list(hyb_cfg.get("current_only", [0.1, 0.15, 0.35, 0.05, 0.35]))
        elif embed_scores or als_scores:
            w = list(hyb_cfg.get("embed_or_als", [0.15, 0.15, 0.1, 0.15, 0.45]))
        else:
            w = list(hyb_cfg.get("cold_start", [0.6, 0.2, 0.2, 0.0, 0.0]))  # cold-start

        if not als_scores:
            w[4] = 0.0
        if not embed_scores:
            w[3] = 0.0
        combined = _combine_scores(content_scores, co_scores, markov_scores, embed_scores, als_scores, weights=w)
        if not combined and markov_scores:
            combined = markov_scores

    if not combined:
        # Fallback por popularidad dentro de la ciudad cargada
        pop = pois_df.copy()
        pop["rating"] = pd.to_numeric(pop["rating"], errors="coerce").fillna(0)
        pop["total_ratings"] = pd.to_numeric(pop.get("total_ratings"), errors="coerce").fillna(0)
        pop["r_norm"] = pop["rating"] / (pop["rating"].max() + 1e-8) if pop["rating"].max() > 0 else 0
        pop["tr_norm"] = (
            np.log1p(pop["total_ratings"]) / (np.log1p(pop["total_ratings"].max()) + 1e-8)
            if pop["total_ratings"].max() > 0
            else 0
        )
        pop["score"] = 0.7 * pop["r_norm"] + 0.3 * pop["tr_norm"]
        pop = pop.sort_values("score", ascending=False).head(k)
        cols = ["fsq_id", "name", "city", "primary_category", "rating", "price_tier", "is_free", "score"]
        return pop[cols]

    # 5) Preparar output y filtros
    candidates = (
        pd.DataFrame({"fsq_id": list(combined.keys()), "score": list(combined.values())})
        .merge(pois_df, on="fsq_id", how="left")
        .sort_values("score", ascending=False)
    )

    # Excluir POI actual
    if current_poi:
        candidates = candidates[candidates["fsq_id"] != current_poi]

    # Eliminar candidatos sin metadatos mínimos
    candidates = candidates.dropna(subset=["name", "primary_category", "lat", "lon"], how="any")

    # Filtrar categorías no deseadas
    if not candidates.empty:
        candidates = candidates[~candidates["primary_category"].isin(EXCLUDE_CATEGORIES)]

    # Preferences can set/override filters.
    if prefs is not None:
        if prefs.free_only is True:
            free_only = True
        elif prefs.free_only is False:
            # Only paid
            free_only = False
        if prefs.max_price_tier is not None:
            max_price_tier = prefs.max_price_tier

    if prefs is not None and prefs.free_only is False:
        candidates = candidates[candidates["is_free"] == False]  # noqa: E712
    elif free_only:
        candidates = candidates[candidates["is_free"] == True]  # noqa: E712
    if max_price_tier is not None:
        candidates["price_tier"] = pd.to_numeric(candidates["price_tier"], errors="coerce")
        candidates = candidates[(candidates["price_tier"].isna()) | (candidates["price_tier"] <= max_price_tier)]

    # Category preference boost (soft, not a hard filter).
    if prefs is not None and prefs.categories and not candidates.empty:
        pref_cats = set([c.lower() for c in normalize_categories(prefs.categories)])
        if pref_cats:
            # Build a per-POI set of category names (including primary_category).
            cats_by_id = (
                poi_cats_df.groupby("fsq_id")["category_name"].apply(lambda s: set([str(x).lower() for x in s.dropna().tolist()]))
                if not poi_cats_df.empty
                else pd.Series(dtype=object)
            )

            def _matches(fid: str, primary: Optional[str]) -> bool:
                s = set()
                if primary:
                    s.add(str(primary).lower())
                extra = cats_by_id.get(fid)
                if isinstance(extra, set):
                    s |= extra
                return bool(s & pref_cats)

            boost = float(prefs_cfg.get("category_boost", 0.2))
            candidates["pref_match"] = candidates.apply(
                lambda r: _matches(str(r["fsq_id"]), r.get("primary_category")),
                axis=1,
            )
            candidates.loc[candidates["pref_match"] == True, "score"] = candidates.loc[  # noqa: E712
                candidates["pref_match"] == True, "score"
            ] * (1.0 + boost)

    # Re-ranking por distancia SOLO si hay ubicacion explicita (lat/lon).
    # Si no se proporciona, dejamos que el ranking lo dominen las senales del modelo.
    if lat is not None and lon is not None and not candidates.empty:
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        candidates["distance_km"] = haversine(lat, lon, candidates["lat"].astype(float), candidates["lon"].astype(float))
        if candidates["distance_km"].max() > 0:
            candidates["dist_norm"] = candidates["distance_km"] / (candidates["distance_km"].max() + 1e-8)
            candidates["score"] = candidates["score"] * (1 - distance_weight * candidates["dist_norm"])

    # Diversidad básica por categoría (si se solicita)
    if diversify and not candidates.empty:
        max_per_cat = int(route_pl_cfg.get("max_per_category", 2))
        cat_counts: Dict[str, int] = {}
        diversified = []
        for _, row in candidates.sort_values("score", ascending=False).iterrows():
            cat = row.get("primary_category")
            if pd.isna(row.get("name")) or pd.isna(cat):
                continue
            cat = str(cat)
            if cat_counts.get(cat, 0) >= max_per_cat:
                continue
            diversified.append(row)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if len(diversified) >= k:
                break
        if diversified:
            candidates = pd.DataFrame(diversified)
        else:
            candidates = candidates.sort_values("score", ascending=False)

    candidates = candidates.sort_values("score", ascending=False).head(k)

    # Asegurar que lat/lon están presentes
    if "lat" not in candidates.columns or "lon" not in candidates.columns:
        candidates = candidates.merge(pois_df[["fsq_id", "lat", "lon"]], on="fsq_id", how="left")

    cols = ["fsq_id", "name", "city", "primary_category", "rating", "price_tier", "is_free", "score", "lat", "lon"]
    if "distance_km" in candidates:
        cols.append("distance_km")
    return candidates[cols]


__all__ = ["recommend"]
