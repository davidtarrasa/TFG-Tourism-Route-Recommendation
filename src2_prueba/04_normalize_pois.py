"""04_normalize_pois.py: fusiona POIs del profesor y API nueva en un esquema canonico."""
import argparse
from pathlib import Path

import pandas as pd

from utils import (
    get_city_config,
    load_json_list,
    normalize_poi_obj,
    merge_pois,
    save_json,
    ensure_latlon,
)

STD_PATH = Path("data/processed/std_clean.csv")


def main():
    parser = argparse.ArgumentParser(description="Normaliza y fusiona POIs (profesor + API)")
    parser.add_argument("--city", required=True, help="osaka / istanbul / petalingjaya")
    parser.add_argument("--prof", help="JSON del profesor")
    parser.add_argument("--api", help="JSON descargado de la API")
    parser.add_argument("--out", help="Salida JSON normalizada")
    args = parser.parse_args()

    cfg = get_city_config(args.city)
    prof_path = args.prof or f"data/raw/ALL_POIS_{cfg['file']}.json"
    api_path = args.api or f"data/raw/raw_pois_{cfg['file']}.json"
    out_path = args.out or f"data/intermediate/pois_norm_{cfg['file']}.json"

    prof_raw = load_json_list(prof_path)
    api_raw = load_json_list(api_path)

    prof_norm = [normalize_poi_obj(o) for o in prof_raw]
    api_norm = [normalize_poi_obj(o) for o in api_raw]

    merged = merge_pois(prof_norm, api_norm)
    cleaned = [p for p in merged.values() if ensure_latlon(p.get("lat"), p.get("lon")) and p.get("categories")]

    # Rellenar city/country si vienen vacíos con la info de cfg
    for p in cleaned:
        if not (p.get("city") or "").strip():
            p["city"] = cfg["name"]
        if not (p.get("country") or "").strip():
            p["country"] = cfg["country"]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_json(cleaned, out_path)
    print(f"[4/8] Guardado {out_path} ({len(cleaned):,} POIs)")

    # Ajustar std_clean: eliminar filas de esta ciudad cuyos venue_id no estén en los POIs válidos
    if STD_PATH.exists():
        std_df = pd.read_csv(STD_PATH)
        before = len(std_df)
        valid_ids = {p.get("fsq_id") for p in cleaned if p.get("fsq_id")}
        mask_city = std_df["venue_city"] == cfg["qid"]
        mask_invalid = ~std_df["venue_id"].astype(str).isin(valid_ids)
        std_df = std_df[~(mask_city & mask_invalid)].copy()
        removed = before - len(std_df)
        std_df.to_csv(STD_PATH, index=False)
        if removed > 0:
            print(f"[4/8] Pruned {removed:,} filas de std_clean.csv para {cfg['name']} (IDs sin categoria/coords)")


if __name__ == "__main__":
    main()
