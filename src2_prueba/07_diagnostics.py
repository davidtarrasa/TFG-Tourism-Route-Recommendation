"""07_diagnostics.py: cobertura std vs POIs y reportes básicos."""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from utils import get_city_config, load_json_list

STD_CLEAN = Path("data/processed/std_clean.csv")
REPORT_DIR = Path("data/reports/diagnostics2")


def coverage(std_df: pd.DataFrame, city_key: str, pois_path: Path):
    cfg = get_city_config(city_key)
    ids_std = set(std_df[std_df["venue_city"] == cfg["qid"]]["venue_id"].astype(str))
    pois = load_json_list(str(pois_path))
    ids_poi = {str(p.get("fsq_id")) for p in pois if p.get("fsq_id")}
    missing = sorted(ids_std - ids_poi)
    return ids_std, ids_poi, missing


def main():
    parser = argparse.ArgumentParser(description="Genera diagnósticos de cobertura std vs POIs")
    parser.add_argument("--cities", nargs="*", default=["osaka", "istanbul", "petalingjaya"])
    parser.add_argument("--pois-pattern", dest="pois_pattern", default="data/processed/pois_enriched_{name}.json")
    args = parser.parse_args()

    df = pd.read_csv(STD_CLEAN, usecols=["venue_id", "venue_city"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for city in args.cities:
        cfg = get_city_config(city)
        pois_path = Path(args.pois_pattern.format(name=cfg["file"]))
        ids_std, ids_poi, missing = coverage(df, city, pois_path)

        (REPORT_DIR / f"missing_{cfg['file']}.txt").write_text("\n".join(missing), encoding="utf-8")
        (REPORT_DIR / f"ids_std_{cfg['file']}.txt").write_text("\n".join(sorted(ids_std)), encoding="utf-8")
        (REPORT_DIR / f"ids_pois_{cfg['file']}.txt").write_text("\n".join(sorted(ids_poi)), encoding="utf-8")

        print(f"[7/8] {cfg['name']}: std={len(ids_std):,} pois={len(ids_poi):,} missing={len(missing):,}")

    counts = Counter(df["venue_city"].astype(str))
    (REPORT_DIR / "std_city_distribution.csv").write_text(
        "venue_city,count\n" + "\n".join(f"{k},{v}" for k, v in counts.items()),
        encoding="utf-8",
    )
    print(f"[7/8] Reportes en {REPORT_DIR}")


if __name__ == "__main__":
    main()
