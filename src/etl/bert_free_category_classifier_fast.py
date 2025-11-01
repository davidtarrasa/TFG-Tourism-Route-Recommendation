"""
Versión rápida (GPU batching) del clasificador de categorías FREE/PAID + TIER.

Variables de entorno soportadas:
- FREE_THRESHOLD (por defecto 0.5)
- PAID_THRESHOLD (por defecto 0.65)
- MARGIN_MIN (por defecto 0.10)
- USE_OBSERVED_FREEPAID (true/false, por defecto true)
- BERT_BATCH_SIZE (tamaño de lote de categorías, por defecto 64)
- BERT_INFER_BATCH (tamaño interno de inferencia del pipeline, por defecto 32)
"""

import os, json, glob, statistics
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import pipeline

# Paths iguales a la versión original
JSON_GLOB = "data/processed/foursquare/ALL_POIS_*_prof_filtered.json"
OUT_JSON  = "data/processed/bert_category_price_labels.json"
OUT_CSV   = "data/reports/bert_category_price_labels.csv"

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
os.makedirs(os.path.dirname(OUT_CSV),  exist_ok=True)

# Modelos
PRIMARY_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"   # multilingüe
FALLBACK_MODEL = "typeform/distilbert-base-uncased-mnli"    # inglés (sin sentencepiece)


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    val = (os.getenv(name, str(default))).strip().lower()
    return val in ("1", "true", "yes", "y", "t")


FREE_THRESHOLD = _get_float_env("FREE_THRESHOLD", 0.55)
PAID_THRESHOLD = _get_float_env("PAID_THRESHOLD", 0.6)
MARGIN_MIN    = _get_float_env("MARGIN_MIN", 0.05)
USE_OBSERVED_FREEPAID = _get_bool_env("USE_OBSERVED_FREEPAID", True)
BATCH = int(os.getenv("BERT_BATCH_SIZE", "64"))
INFER_BATCH = int(os.getenv("BERT_INFER_BATCH", "32"))
NAME_RULE_PAID_SUBSTRINGS = {"restaurant", "pub", "service"}


FREEPAID_LABELS = [
    "free (public access, no mandatory payment)",
    "paid (requires payment to access/use or consume)",
]
TIER_LABELS = [
    "cheap (tier 1)",
    "moderate (tier 2)",
    "expensive (tier 3)",
    "very expensive (tier 4)",
]
TEMPLATES = [
    "Este lugar es {}.",
    "Esta categoría normalmente es {}.",
    "People can use or access this category and it is {}.",
    "This category is typically {}.",
    "This place can be used without paying money, so it is {}.",
]


FREE_GT_FORCE = {
    "12094", "12064", "12000", "11045", "11128", "11001", "12047",
    "12009", "12080", "12013", "12014", "12031", "12022", "12021", "12048", "12058",
    "12017", "12101", "12106", "12108", "12099", "16032", "16041", "16020", "16026",
    "16017", "16046", "16001", "16006", "16007", "16030", "12075", "16003", "15014",
    "10047", "12003", "19042", "19046", "19050", "19022", "19047", "12086",
}
PAID_GT_FORCE = {
    "13065","13034","13003","13020","13035","13099","13272","13026","17029","13145",
    "17043","10032","17035","13064","13006","10021","17018","17048","19014","17036"
}


def get_classifier():
    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline(
            "zero-shot-classification",
            model=PRIMARY_MODEL,
            tokenizer=PRIMARY_MODEL,
            use_fast=False,
            device=device,
        ), PRIMARY_MODEL
    except Exception as e:
        print(f"[WARN] Multilingüe no disponible ({e}). Uso fallback inglés DistilBERT-MNLI.")
        return pipeline(
            "zero-shot-classification",
            model=FALLBACK_MODEL,
            tokenizer=FALLBACK_MODEL,
            use_fast=False,
            device=device,
        ), FALLBACK_MODEL


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


def batch_agg_probs(clf, texts, labels, templates, batch_size=32):
    if len(texts) == 0:
        return []
    acc = [{l: 0.0 for l in labels} for _ in texts]
    for tmpl in templates:
        outs = clf(
            texts,
            labels,
            hypothesis_template=tmpl.format("{}"),
            batch_size=batch_size,
            truncation=True,
        )
        for i, o in enumerate(outs):
            for lab, sc in zip(o["labels"], o["scores"]):
                acc[i][lab] += float(sc)
    for d in acc:
        for l in d:
            d[l] /= len(templates)
    return acc


def batch_decide_freepaid(cids, names, clf):
    out = []
    override_mask = []
    for cid in cids:
        if cid in FREE_GT_FORCE:
            out.append((True, 1.0, "override_free"))
            override_mask.append(True)
        elif cid in PAID_GT_FORCE:
            out.append((False, 1.0, "override_paid"))
            override_mask.append(True)
        else:
            out.append(None)
            override_mask.append(False)
    need_idx = [i for i, m in enumerate(override_mask) if not m]
    need_texts = [names[i] for i in need_idx]
    if need_idx:
        scores = batch_agg_probs(clf, need_texts, FREEPAID_LABELS, TEMPLATES, batch_size=INFER_BATCH)
        for local_i, scmap in enumerate(scores):
            i = need_idx[local_i]
            ordered = sorted(scmap.items(), key=lambda x: x[1], reverse=True)
            (best_lab, best_sc), (_, second_sc) = ordered[0], ordered[1]
            margin = best_sc - second_sc
            is_free = ("free" in best_lab)
            if is_free and best_sc >= FREE_THRESHOLD and margin >= MARGIN_MIN:
                out[i] = (True, best_sc, "model_confident")
            elif (not is_free) and best_sc >= PAID_THRESHOLD and margin >= MARGIN_MIN:
                out[i] = (False, best_sc, "model_confident")
            else:
                out[i] = (is_free, best_sc, "model_uncertain")
    return out


def batch_decide_tier_bert(names, clf):
    mapping = {
        "cheap (tier 1)": 1,
        "moderate (tier 2)": 2,
        "expensive (tier 3)": 3,
        "very expensive (tier 4)": 4,
    }
    scores = batch_agg_probs(clf, names, TIER_LABELS, TEMPLATES, batch_size=INFER_BATCH)
    out = []
    for scmap in scores:
        best_lab, best_sc = max(scmap.items(), key=lambda x: x[1])
        out.append((mapping[best_lab], best_sc))
    return out


def load_categories_and_prices():
    cat_name = {}
    cat_prices = defaultdict(list)
    files = glob.glob(JSON_GLOB)
    if not files:
        print(f"[WARN] No se han encontrado ficheros con el patron: {JSON_GLOB}")
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for poi in data:
            cats = poi.get("categories") or []
            price = poi.get("price")
            for c in cats:
                cid, cname = c.get("id"), c.get("name")
                if not cid or not cname:
                    continue
                if cid not in cat_name:
                    cat_name[cid] = cname
                if isinstance(price, (int, float)):
                    cat_prices[cid].append(int(price))
    return cat_name, cat_prices


def decide_tier_from_observed(prices):
    paid_vals = [p for p in prices if isinstance(p, int) and p > 0 and 1 <= p <= 4]
    if len(paid_vals) >= 3:
        med = int(round(statistics.median(paid_vals)))
        return max(1, min(4, med))
    return None


def observed_free_from_prices(prices):
    vals = [p for p in prices if isinstance(p, int) and 0 <= p <= 4]
    if len(vals) == 0:
        return None
    med = statistics.median(vals)
    return (med == 0)


def main():
    clf, used_model = get_classifier()
    try:
        print(f"Dispositivo del modelo: {clf.model.device}")
    except Exception:
        pass
    print(f"Modelo usado: {used_model}")
    print(
        f"Umbrales -> FREE_THRESHOLD={FREE_THRESHOLD}, PAID_THRESHOLD={PAID_THRESHOLD}, MARGIN_MIN={MARGIN_MIN}"
    )
    print(f"USE_OBSERVED_FREEPAID={USE_OBSERVED_FREEPAID}")

    cat_name, cat_prices = load_categories_and_prices()
    print(f"Categorías únicas: {len(cat_name):,}")

    items = list(cat_name.items())
    results = {}
    with open(OUT_CSV, "w", encoding="utf-8") as fcsv:
        fcsv.write("category_id,name,free,price_tier,source,score,model\n")
        for chunk in tqdm(list(chunked(items, BATCH)), desc="Clasificando categorías"):
            cids = [cid for cid, _ in chunk]
            names = [name for _, name in chunk]

            if USE_OBSERVED_FREEPAID:
                observed_flags = [observed_free_from_prices(cat_prices.get(cid, [])) for cid in cids]
            else:
                observed_flags = [None] * len(cids)

            # Regla por nombre: si contiene restaurant/pub/service => paid
            pre_fp = [None] * len(cids)
            for i, nm in enumerate(names):
                nm_l = (nm or "").lower()
                if any(sub in nm_l for sub in NAME_RULE_PAID_SUBSTRINGS):
                    pre_fp[i] = (False, 1.0, "name_rule_paid")

            need_idx = [i for i, flag in enumerate(observed_flags) if flag is None]
            fp = [None] * len(cids)
            if need_idx:
                fp_sub = batch_decide_freepaid(
                    [cids[i] for i in need_idx],
                    [names[i] for i in need_idx],
                    clf,
                )
                for j, i in enumerate(need_idx):
                    fp[i] = fp_sub[j]

            for i in range(len(cids)):
                if observed_flags[i] is not None and fp[i] is None and pre_fp[i] is None:
                    is_free = bool(observed_flags[i])
                    fp[i] = (is_free, 1.0, "observed_free" if is_free else "observed_paid")

            # Prioridad final: regla por nombre
            for i in range(len(cids)):
                if pre_fp[i] is not None:
                    fp[i] = pre_fp[i]

            need_tier_names = []
            need_tier_idx = []
            for i, cid in enumerate(cids):
                is_free, fscore, reason = fp[i]
                if is_free:
                    results[cid] = {
                        "name": names[i],
                        "free": True,
                        "price_tier": 0,
                        "source": reason,
                        "score": round(fscore, 4),
                        "model": used_model,
                    }
                    fcsv.write(
                        f"{cid},{names[i]},1,0,{reason},{round(fscore,4)},{used_model}\n"
                    )
                else:
                    tier = decide_tier_from_observed(cat_prices.get(cid, []))
                    if tier is not None:
                        results[cid] = {
                            "name": names[i],
                            "free": False,
                            "price_tier": tier,
                            "source": "observed_median",
                            "score": "",
                            "model": used_model,
                        }
                        fcsv.write(
                            f"{cid},{names[i]},0,{tier},observed_median,,{used_model}\n"
                        )
                    else:
                        need_tier_names.append(names[i])
                        need_tier_idx.append(i)

            if need_tier_names:
                tiers = batch_decide_tier_bert(need_tier_names, clf)
                for (tier, tscore), i in zip(tiers, need_tier_idx):
                    cid = cids[i]
                    results[cid] = {
                        "name": names[i],
                        "free": False,
                        "price_tier": tier,
                        "source": "bert",
                        "score": round(tscore, 4),
                        "model": used_model,
                    }
                    fcsv.write(
                        f"{cid},{names[i]},0,{tier},bert,{round(tscore,4)},{used_model}\n"
                    )

    with open(OUT_JSON, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f"\n✅ Guardado JSON: {OUT_JSON}")
    print(f"🗂️  Guardado CSV:  {OUT_CSV}")


if __name__ == "__main__":
    main()
