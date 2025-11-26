"""02_extract_ids.py: extrae ids de POI por ciudad desde std_clean."""
import argparse
from pathlib import Path

import pandas as pd

from utils import get_city_config, clean_fsq_id

STD_CLEAN = Path("data/processed/std_clean.csv")
OUT_DIR = Path("data/intermediate")


def extract_ids(city: str):
    cfg = get_city_config(city)
    df = pd.read_csv(STD_CLEAN, usecols=["venue_id", "venue_city"])
    df_city = df[df["venue_city"] == cfg["qid"]].copy()
    ids = df_city["venue_id"].dropna().astype(str).map(clean_fsq_id).dropna().unique().tolist()

    out_path = OUT_DIR / f"ids_{cfg['file']}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for vid in ids:
            f.write(f"{vid}\n")
    print(f"[2/8] {cfg['name']}: {len(ids):,} ids -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extrae IDs de POIs por ciudad desde std_clean.csv")
    parser.add_argument("--city", required=True, help="osaka / istanbul / petalingjaya")
    args = parser.parse_args()
    extract_ids(args.city)


if __name__ == "__main__":
    main()
