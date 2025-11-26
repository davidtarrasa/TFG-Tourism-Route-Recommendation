"""01_clean_std.py: limpia std_2018 y genera std_clean.csv."""
import pandas as pd
from pathlib import Path

from utils import clean_prefix

STD_PATH = Path("data/raw/std_2018.csv")
OUT_PATH = Path("data/processed/std_clean.csv")
VALID_CITIES_RAW = ["wd:Q35765", "wd:Q406", "wd:Q864965"]


def normalize_field(value: str) -> str:
    """Quita prefijos conocidos y espacios."""
    if not isinstance(value, str):
        return value
    value = clean_prefix(value, "foursquare:")
    value = clean_prefix(value, "schema:")
    value = clean_prefix(value, "wd:")
    return value.strip()


def reindex_to_start_at_1(series: pd.Series) -> pd.Series:
    """Mapea valores únicos a 1..N preservando orden de aparición."""
    uniques = pd.Index(series.dropna().unique())
    mapping = {old: i + 1 for i, old in enumerate(uniques)}
    return series.map(mapping)


def main():
    print("[1/8] Leyendo std_2018...")
    df = pd.read_csv(STD_PATH)
    df = df[df["venue_city"].isin(VALID_CITIES_RAW)].copy()

    df = df.dropna(subset=["venue_id", "timestamp"])
    for col in ["venue_id", "venue_category", "venue_schema", "venue_city", "venue_country"]:
        df[col] = df[col].astype(str).str.strip().map(normalize_field)

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.loc[~ts.isna()].copy()
    df["timestamp"] = ts.loc[~ts.isna()].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if "trail_id" in df.columns:
        df["trail_id_orig"] = df["trail_id"]
        df["trail_id"] = reindex_to_start_at_1(df["trail_id_orig"]).astype(int)
    if "user_id" in df.columns:
        df["user_id_orig"] = df["user_id"]
        df["user_id"] = reindex_to_start_at_1(df["user_id_orig"]).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[1/8] Guardado {OUT_PATH} ({len(df):,} filas)")


if __name__ == "__main__":
    main()
