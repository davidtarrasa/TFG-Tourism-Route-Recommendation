# src/etl/bert_price_classifier.py
"""
Clasifica categorías Foursquare como FREE/PAID y, si PAID, asigna price_tier (1..4).
Salida: data/processed/bert_category_price_labels.json

Lógica:
- FREE vs PAID con zero-shot (multilingüe estable, Windows-friendly).
- Si PAID, intenta usar la mediana observada por categoría (de los JSON) como tier.
- Si no hay datos suficientes, usa zero-shot para tier:
    ["cheap (tier 1)", "moderate (tier 2)", "expensive (tier 3)", "very expensive (tier 4)"]
- Umbrales y overrides mínimos para estabilidad.

Requisitos:
  pip install transformers torch sentencepiece tqdm
(Si no quieres sentencepiece, usa el fallback inglés como indico abajo.)
"""

import os, json, glob, statistics
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import pipeline

# ---------------- Paths ----------------
JSON_GLOB = "data/processed/foursquare/ALL_POIS_*_prof_filtered.json"
OUT_JSON  = "data/processed/bert_category_price_labels.json"
OUT_CSV   = "data/reports/bert_category_price_labels.csv"

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
os.makedirs(os.path.dirname(OUT_CSV),  exist_ok=True)

# ---------------- Modelo ----------------
PRIMARY_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"   # multilingüe
FALLBACK_MODEL = "typeform/distilbert-base-uncased-mnli"    # sin sentencepiece (inglés)

def get_classifier():
    # Selección automática de dispositivo (GPU si disponible)
    device = 0 if torch.cuda.is_available() else -1
    # intenta multilingüe
    try:
        return pipeline(
            "zero-shot-classification",
            model=PRIMARY_MODEL, tokenizer=PRIMARY_MODEL,
            use_fast=False, device=0
        ), PRIMARY_MODEL
    except Exception as e:
        print(f"[WARN] Multilingüe no disponible ({e}). Uso fallback inglés DistilBERT-MNLI.")
        return pipeline(
            "zero-shot-classification",
            model=FALLBACK_MODEL, tokenizer=FALLBACK_MODEL,
            use_fast=False, device=device
        ), FALLBACK_MODEL

# ------------- Labels / templates -------------
FREEPAID_LABELS = [
    "free (public access, no mandatory payment)",
    "paid (requires payment to access/use or consume)"
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
FREE_THRESHOLD = 0.5
PAID_THRESHOLD = 0.65
MARGIN_MIN    = 0.10

# --------- Overrides simples (ajusta si quieres) ---------
FREE_GT_FORCE = {
    "11124",  # Office
    "11130",  # Office Building
    "12094",  # Residential Building
    "12064",  # Government Building
    "12000",  # Community and Government (macro)
    "11045",  # Bank
    "11128",  # Coworking Space
    "11001",  # Advertising Agency
    "12047",  # Student Center
    "12009",  # Education
    "12080",  # Library
    "12013",  # College and University
    "12014",  # College Academic Building
    "12031",  # College Library
    "12022",  # College Classroom
    "12021",  # College Cafeteria (suele ser acceso libre, gasto opcional)
    "12048",  # Community College
    "12058",  # Elementary School
    "12017",  # College auditorium
    "12101",  # Church
    "12106",  # Mosque
    "12108",  # Shrine
    "12099",  # Buddhist Temple
    "16032",  # Park
    "16041",  # Plaza
    "16020",  # Historic and Protected Site
    "16026",  # Monument
    "16017",  # Garden
    "16046",  # Scenic Lookout
    "16001",  # Bathing Area
    "16006",  # Bridge
    "16007",  # Structure (genérico)
    "16030",  # Other Great Outdoors
    "12075",  # Post Office
    "16003",  # Beach
    "15014",  # Hospital
    "10047",  # Public Art
    "12003",  # Cemetery
    "19042",  # Bus Station
    "19046",  # Metro Station
    "19050",  # Tram Station
    "19022",  # Platform
    "19047",  # Rail Station
    "12086",
}
PAID_GT_FORCE = {
    "13065","13034","13003","13020","13035","13099","13272","13026","17029","13145",
    "17043","10032","17035","13064","13006","10021","17018","17048","19014","17036"
}

# --------- Cargar categorías y precios observados ---------
def load_categories_and_prices():
    cat_name = {}                   # id -> name
    cat_prices = defaultdict(list)  # id -> [price tiers 0..4]
    for path in glob.glob(JSON_GLOB):
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
                    # guardamos todas (incluye 0=gratis)
                    cat_prices[cid].append(int(price))
    return cat_name, cat_prices

def agg_probs(clf, text, labels):
    agg = {l: 0.0 for l in labels}
    for tmpl in TEMPLATES:
        pred = clf(text, labels, hypothesis_template=tmpl.format("{}"))
        for lab, sc in zip(pred["labels"], pred["scores"]):
            agg[lab] += float(sc)
    # promedio
    for l in agg: agg[l] /= len(TEMPLATES)
    ordered = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return ordered  # list of (label, score) sorted desc

def decide_freepaid(cid, name, clf):
    # Overrides primero
    if cid in FREE_GT_FORCE:
        return True, 1.0, "override_free"
    if cid in PAID_GT_FORCE:
        return False, 1.0, "override_paid"

    ordered = agg_probs(clf, name, FREEPAID_LABELS)
    (best_lab, best_sc), (_, second_sc) = ordered[0], ordered[1]
    margin = best_sc - second_sc
    is_free = ("free" in best_lab)

    if is_free and best_sc >= FREE_THRESHOLD and margin >= MARGIN_MIN:
        return True, best_sc, "model_confident"
    if (not is_free) and best_sc >= PAID_THRESHOLD and margin >= MARGIN_MIN:
        return False, best_sc, "model_confident"
    # incierto -> nos quedamos con la predicción, pero marcamos reason
    return is_free, best_sc, "model_uncertain"

def decide_tier_from_observed(prices):
    # usa mediana de precios >0; si no hay suficientes, devuelve None
    paid_vals = [p for p in prices if isinstance(p, int) and p > 0 and 1 <= p <= 4]
    if len(paid_vals) >= 3:
        # mediana y clip 1..4
        med = int(round(statistics.median(paid_vals)))
        return max(1, min(4, med))
    return None

def decide_tier_from_bert(name, clf):
    ordered = agg_probs(clf, name, TIER_LABELS)
    # map label -> tier
    mapping = {
        "cheap (tier 1)": 1,
        "moderate (tier 2)": 2,
        "expensive (tier 3)": 3,
        "very expensive (tier 4)": 4
    }
    best_lab, best_sc = ordered[0]
    return mapping[best_lab], best_sc

def main():
    clf, used_model = get_classifier()
    # Log rápido del dispositivo usado por el modelo (cuda:0 si GPU)
    try:
        print(f"Dispositivo del modelo: {clf.model.device}")
    except Exception:
        pass
    cat_name, cat_prices = load_categories_and_prices()
    print(f"Categorías únicas: {len(cat_name):,}")

    results = {}
    with open(OUT_CSV, "w", encoding="utf-8") as fcsv:
        fcsv.write("category_id,name,free,price_tier,source,score,model\n")

        for cid, name in tqdm(cat_name.items(), desc="Clasificando categorías"):
            # 1) FREE vs PAID
            is_free, fscore, reason = decide_freepaid(cid, name, clf)
            if is_free:
                results[cid] = {
                    "name": name, "free": True, "price_tier": 0,
                    "source": reason, "score": round(fscore,4), "model": used_model
                }
                fcsv.write(f"{cid},{name},1,0,{reason},{round(fscore,4)},{used_model}\n")
                continue

            # 2) TIER: primero mediana observada; si no hay, BERT
            tier = decide_tier_from_observed(cat_prices.get(cid, []))
            if tier is not None:
                results[cid] = {
                    "name": name, "free": False, "price_tier": tier,
                    "source": "observed_median", "score": "", "model": used_model
                }
                fcsv.write(f"{cid},{name},0,{tier},observed_median,,{used_model}\n")
            else:
                tier, tscore = decide_tier_from_bert(name, clf)
                results[cid] = {
                    "name": name, "free": False, "price_tier": tier,
                    "source": "bert", "score": round(tscore,4), "model": used_model
                }
                fcsv.write(f"{cid},{name},0,{tier},bert,{round(tscore,4)},{used_model}\n")

    with open(OUT_JSON, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado JSON: {OUT_JSON}")
    print(f"🗂️  Guardado CSV:  {OUT_CSV}")

if __name__ == "__main__":
    main()
