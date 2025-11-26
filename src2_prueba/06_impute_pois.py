"""06_impute_pois.py: aplica labels de categoría para imputar precio/gratuidad y limpia ratings."""
import argparse
import json
from pathlib import Path

from utils import load_json_list, save_json, ensure_latlon, safe_int, get_city_config

LABELS_PATH = Path("data/processed/category_price_labels.json")


def apply_labels(pois: list, labels: dict) -> list:
    for poi in pois:
        cats = poi.get("categories") or []
        tier = None
        is_free = False
        for c in cats:
            cid = str(c.get("id")) if c.get("id") is not None else None
            if not cid:
                continue
            info = labels.get(cid)
            if info:
                if info.get("free") is True:
                    tier = 0
                    is_free = True
                    break
                if tier is None and isinstance(info.get("price_tier"), int):
                    tier = info.get("price_tier")
        if tier is not None:
            poi["price"] = tier
        poi["is_free"] = bool(is_free or tier == 0)

        poi["total_ratings"] = safe_int(poi.get("total_ratings")) or 0
        try:
            poi["rating"] = float(poi.get("rating")) if poi.get("rating") not in (None, "", []) else None
        except Exception:
            poi["rating"] = None
    return pois


def main():
    parser = argparse.ArgumentParser(description="Enriquece POIs con labels de categoría")
    parser.add_argument("--city", required=True, help="osaka / istanbul / petalingjaya")
    parser.add_argument("--infile", help="POIs normalizados por ciudad")
    parser.add_argument("--out", help="Salida enriquecida")
    args = parser.parse_args()

    if not LABELS_PATH.exists():
        raise SystemExit("No existe category_price_labels.json. Ejecuta 05_label_categories primero.")
    labels = json.load(LABELS_PATH.open())

    cfg = get_city_config(args.city)
    infile = args.infile or f"data/intermediate/pois_norm_{cfg['file']}.json"
    outfile = args.out or f"data/processed/pois_enriched_{cfg['file']}.json"

    pois = load_json_list(infile)
    pois = [p for p in pois if ensure_latlon(p.get("lat"), p.get("lon"))]
    enriched = apply_labels(pois, labels)

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    save_json(enriched, outfile)
    print(f"[6/8] Guardado {outfile} ({len(enriched):,} POIs)")


if __name__ == "__main__":
    main()
