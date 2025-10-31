# src/etl/bert_price_check_crossval.py
import os, json, glob, statistics
from collections import defaultdict

LABELS_PATH = "data/processed/bert_category_price_labels.json"
JSON_GLOB   = "data/processed/foursquare/ALL_POIS_*_prof_filtered.json"

# --- categorías forzadas (las que el modelo no debe evaluar) ---
FREE_GT_FORCE = {  # mismo set que en el clasificador
    "11124","11130","12094","12064","12000","11045","11128","11001","12047",
    "12009","12080","12013","12014","12031","12022","12021","12048","12058",
    "12017","12101","12106","12108","12099","16032","16041","16020","16026",
    "16017","16046","16001","16006","16007","16030","12075","16003","15014",
    "10047","12003","19042","19046","19050","19022","19047","12086"
}
PAID_GT_FORCE = {
    "13065","13034","13003","13020","13035","13099","13272","13026","17029","13145",
    "17043","10032","17035","13064","13006","10021","17018","17048","19014"
}
FORCED = FREE_GT_FORCE | PAID_GT_FORCE

# --- cargar resultados BERT ---
if not os.path.exists(LABELS_PATH):
    raise SystemExit(f"No existe {LABELS_PATH}. Ejecuta primero el clasificador de precio BERT.")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    bert = json.load(f)

# --- obtener ground truth observada desde los POIs ---
cat_prices = defaultdict(list)
cat_names  = {}
for path in glob.glob(JSON_GLOB):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for poi in data:
        price = poi.get("price")
        cats = poi.get("categories") or []
        for c in cats:
            cid, cname = c.get("id"), c.get("name")
            if not cid: continue
            if cname: cat_names[cid] = cname
            if isinstance(price, (int,float)):
                cat_prices[cid].append(int(price))

# --- construir mediana observada por categoría ---
obs = {}
for cid, vals in cat_prices.items():
    vals = [v for v in vals if 0 <= v <= 4]
    if not vals: continue
    med = statistics.median(vals)
    obs[cid] = {"median_price": med, "free": (med == 0)}

# --- comparar solo sobre categorías NO forzadas ---
test_ids = [cid for cid in bert if cid not in FORCED and cid in obs]

tp=fp=tn=fn=0
tier_ok = total_tier = 0
diff_sum = 0.0

for cid in test_ids:
    b = bert[cid]; o = obs[cid]
    pred_free = b["free"]; gt_free = o["free"]
    if gt_free and pred_free: tp+=1
    elif gt_free and not pred_free: fn+=1
    elif (not gt_free) and (not pred_free): tn+=1
    elif (not gt_free) and pred_free: fp+=1

    # tier difference if paid
    if not gt_free and not pred_free:
        pred_tier = b.get("price_tier")
        gt_tier   = round(o["median_price"])
        if isinstance(pred_tier,int):
            total_tier+=1
            if pred_tier==gt_tier: tier_ok+=1
            diff_sum += abs(pred_tier-gt_tier)

# --- métricas ---
def safe_div(a,b): return a/b if b else 0.0
precision = safe_div(tp,tp+fp)
recall    = safe_div(tp,tp+fn)
f1        = safe_div(2*precision*recall, precision+recall)
tier_acc  = safe_div(tier_ok,total_tier)
mae       = safe_div(diff_sum,total_tier)

print("\n=== EVALUACIÓN CRUZADA BERT PRICE ===")
print(f"Categorías testadas (no forzadas): {len(test_ids)}")
print(f"FREE/PAID -> Precision:{precision:.3f} Recall:{recall:.3f} F1:{f1:.3f}")
print(f"Tier accuracy (pagados): {tier_acc:.3f} | MAE abs diff: {mae:.3f}")
