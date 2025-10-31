# src/etl/bert_price_check.py
import os, json, csv

LABELS_PATH = "data/processed/bert_category_price_labels.json"
OUT_DIR = "data/reports/bert_eval"
os.makedirs(OUT_DIR, exist_ok=True)

# GT FREE / PAID (lo de antes)
FREE_GT = {
    "11124",  # Office
    "11130",  # Office Building
    "12094",  # Residential Building
    "12064",  # Government Building
    "12000",  # Community and Government (macro)
    "11045",  # Bank
    "11128",  # Coworking Space
    "11001",  # Advertising Agency
    "12047",  # Student Center
    # Educación (normalmente acceso libre a zonas comunes)
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
    # Religioso
    "12101",  # Church
    "12106",  # Mosque
    "12108",  # Shrine
    "12099",  # Buddhist Temple
    # Exteriores / Espacios públicos
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
    # Arte público
    "15014",  # Hospital
    "10047",  # Public Art
    # Cementerio
    "12003",  # Cemetery
    # Transporte (acceso a la infraestructura suele ser libre; gasto es opcional)
    "19042",  # Bus Station
    "19046",  # Metro Station
    "19050",  # Tram Station
    "19022",  # Platform
    "19047",  # Rail Station
    # Otras organizaciones sin ánimo de lucro
    "12086",  # Non-Profit Organization
}
PAID_GT = {
    "13065","13034","13003","13020","13035","13099","13272","13026","17029","13145",
    "17043","10032","17035","13064","13006","10021","17018","17048","19014"
}
GT_ALL = FREE_GT | PAID_GT

# (Opcional) GT de tier esperada para algunas categorías pagadas
PAID_TIERS_GT = {
    # "13065": 2,  # Restaurant ~ tier 2 (si lo quieres comprobar)
    # "13034": 2,  # Café
}

if not os.path.exists(LABELS_PATH):
    raise SystemExit(f"No existe {LABELS_PATH}. Ejecuta primero el clasificador de precio BERT.")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)  # {cat_id: {"name","free":bool,"price_tier":int,...}}

rows = []
for cid in GT_ALL:
    info = labels.get(cid)
    if not info:
        rows.append({"category_id": cid, "name": "(NO EN JSON)", "gt_free": (cid in FREE_GT),
                     "pred_free": None, "pred_tier": None, "tier_ok": ""})
        continue
    pred_free = bool(info.get("free"))
    pred_tier = info.get("price_tier")
    tier_ok = ""
    if cid in PAID_TIERS_GT and pred_free is False and isinstance(pred_tier, int):
        tier_ok = int(pred_tier) == int(PAID_TIERS_GT[cid])
    rows.append({
        "category_id": cid, "name": info.get("name"), "gt_free": (cid in FREE_GT),
        "pred_free": pred_free, "pred_tier": pred_tier, "tier_ok": tier_ok
    })

# métricas FREE/PAID
tp = fp = tn = fn = 0
for r in rows:
    gt = r["gt_free"]
    pred = r["pred_free"]
    if pred is None: continue
    if gt and pred: tp += 1
    elif gt and not pred: fn += 1
    elif (not gt) and (not pred): tn += 1
    elif (not gt) and pred: fp += 1

def safe_div(a,b): return a/b if b else 0.0
precision = safe_div(tp, tp+fp)
recall    = safe_div(tp, tp+fn)
f1        = safe_div(2*precision*recall, precision+recall)

print("\n=== EVALUACIÓN BERT PRICE (FREE vs PAID) ===")
print(f"Total GT evaluadas: {len(rows)}")
print(f"TP:{tp} FP:{fp} FN:{fn} TN:{tn}")
print(f"Precision: {precision:.3f} Recall: {recall:.3f} F1:{f1:.3f}")

# tier accuracy (solo en las categorías con GT de tier)
tier_rows = [r for r in rows if r["tier_ok"] != ""]
if tier_rows:
    tier_acc = safe_div(sum(1 for r in tier_rows if r["tier_ok"]), len(tier_rows))
    print(f"\nTier accuracy (en {len(tier_rows)} categorías con GT): {tier_acc:.3f}")

# CSV
with open(os.path.join(OUT_DIR, "bert_price_eval_rows.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["category_id","name","gt_free","pred_free","pred_tier","tier_ok"])
    w.writeheader(); w.writerows(rows)

print(f"\n🗂️  CSV guardado en {OUT_DIR}")
