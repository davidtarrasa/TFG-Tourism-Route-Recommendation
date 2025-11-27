# -*- coding: utf-8 -*-
"""06_impute_pois.py: aplica labels de categoría para imputar precio/gratuidad y limpia ratings."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from utils import load_json_list, save_json, ensure_latlon, safe_int, get_city_config

LABELS_PATH = Path("data/processed/category_price_labels.json")


def build_category_medians(pois: list):
    price_by_cat = defaultdict(list)
    rating_by_cat = defaultdict(list)
    tr_by_cat = defaultdict(list)
    for poi in pois:
        cats = poi.get("categories") or []
        price = poi.get("price")
        rating = poi.get("rating")
        tr = poi.get("total_ratings")
        for c in cats:
            cid = str(c.get("id")) if c.get("id") is not None else None
            if not cid:
                continue
            if isinstance(price, (int, float)):
                price_by_cat[cid].append(price)
            if isinstance(rating, (int, float)):
                rating_by_cat[cid].append(rating)
            if isinstance(tr, (int, float)):
                tr_by_cat[cid].append(tr)

    med_price, med_rating, med_tr = {}, {}, {}
    for cid, vals in price_by_cat.items():
        vals = [v for v in vals if 0 <= v <= 4]
        if vals:
            med_price[cid] = statistics.median(vals)
    for cid, vals in rating_by_cat.items():
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            med_rating[cid] = statistics.median(vals)
    for cid, vals in tr_by_cat.items():
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            med_tr[cid] = statistics.median(vals)
    return med_price, med_rating, med_tr


def build_label_indexes(labels: dict):
    name_to_label = defaultdict(list)
    for cid, info in labels.items():
        cname = (info.get("name") or "").strip()
        if cname:
            name_to_label[cname.lower()].append(info)
    return name_to_label


def apply_labels_and_impute(pois: list, labels: dict) -> list:
    med_price, med_rating, med_tr = build_category_medians(pois)
    name_to_label = build_label_indexes(labels)

    for poi in pois:
        cats = poi.get("categories") or []
        cat_ids = [str(c.get("id")) for c in cats if c.get("id") is not None]
        cat_names = [(c.get("name") or "").strip().lower() for c in cats if c.get("name")]

        # Precio via labels BERT (id); si no, por nombre
        tier = None
        is_free = False
        for cid in cat_ids:
            info = labels.get(cid)
            if not info:
                continue
            if info.get("free") is True:
                tier = 0
                is_free = True
                break
            if tier is None and isinstance(info.get("price_tier"), int):
                tier = info.get("price_tier")
        if tier is None:
            for cname in cat_names:
                infos = name_to_label.get(cname, [])
                for info in infos:
                    if info.get("free") is True:
                        tier = 0
                        is_free = True
                        break
                    if tier is None and isinstance(info.get("price_tier"), int):
                        tier = info.get("price_tier")
                if tier is not None:
                    break

        # Si no hubo label, intentar mediana observada por categoría
        if tier is None:
            for cid in cat_ids:
                if cid in med_price:
                    tier = int(round(med_price[cid]))
                    break

        if tier is not None:
            poi["price"] = tier
        poi["is_free"] = bool(is_free or tier == 0)

        # Rating
        if poi.get("rating") in (None, "", []):
            for cid in cat_ids:
                if cid in med_rating:
                    poi["rating"] = float(med_rating[cid])
                    break

        # Total ratings
        if poi.get("total_ratings") in (None, "", []):
            for cid in cat_ids:
                if cid in med_tr:
                    poi["total_ratings"] = int(round(med_tr[cid]))
                    break
            if poi.get("total_ratings") in (None, "", []):
                poi["total_ratings"] = 0

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
    enriched = apply_labels_and_impute(pois, labels)

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    save_json(enriched, outfile)
    print(f"[6/8] Guardado {outfile} ({len(enriched):,} POIs)")


if __name__ == "__main__":
    main()
