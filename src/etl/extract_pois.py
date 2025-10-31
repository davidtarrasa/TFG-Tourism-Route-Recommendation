import pandas as pd
from pathlib import Path

STD_PATH = Path("data/raw/std_2018.csv")
OUT_FILE = Path("data/raw/POIS_osaka_ids.txt")

OSAKA_QID = "wd:Q35765"  # Osaka

def main():
    df = pd.read_csv(STD_PATH)
    # Filtra filas de Osaka
    df_osaka = df[df["venue_city"] == OSAKA_QID].copy()

    # Extrae IDs únicos
    ids = df_osaka["venue_id"].dropna().unique().tolist()

    # Limpia prefijo "foursquare:" si existe
    clean_ids = [i.split("foursquare:")[-1] for i in ids]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for vid in clean_ids:
            f.write(vid.strip() + "\n")

    print(f"[OK] Osaka: {len(clean_ids)} ids -> {OUT_FILE}")

if __name__ == "__main__":
    main()
