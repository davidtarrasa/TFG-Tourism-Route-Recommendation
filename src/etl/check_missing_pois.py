import pandas as pd
import json
from pathlib import Path

# -----------------------------
# Rutas principales
# -----------------------------
STD_CLEAN = Path("data/processed/std_2018_clean.csv")
FOURSQUARE_DIR = Path("data/processed/foursquare")
OUT_ERRORS = Path("data/errors/missing_foursquare_ids.txt")

# -----------------------------
# Ciudades del proyecto
# -----------------------------
CITY_QIDS = {
    "osaka": "Q35765",
    "istanbul": "Q406",
    "petalingjaya": "Q864965"
}

def main():
    print("📥 Cargando dataset limpio...")
    df = pd.read_csv(STD_CLEAN)
    ids_std = set(df["venue_id"].dropna().astype(str))
    print(f"Total de IDs en std_2018_clean.csv: {len(ids_std):,}")

    # -----------------------------
    # Cargar todos los POIs procesados
    # -----------------------------
    poi_files = [
        "ALL_POIS_Istanbul_prof_filtered.json",
        "ALL_POIS_Osaka_prof_filtered.json",
        "ALL_POIS_PetalingJaya_prof_filtered.json"
    ]

    ids_fsq = set()
    for file in poi_files:
        path = FOURSQUARE_DIR / file
        if not path.exists():
            print(f"⚠️  No encontrado: {file}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for obj in data:
                    if isinstance(obj, dict) and "fsq_id" in obj:
                        ids_fsq.add(str(obj["fsq_id"]).strip())
            except Exception as e:
                print(f"⚠️  Error leyendo {file}: {e}")

    print(f"📦 Total de POIs descargados (Foursquare): {len(ids_fsq):,}")

    # -----------------------------
    # Calcular faltantes
    # -----------------------------
    missing_ids = ids_std - ids_fsq
    print(f"❌ Faltan por descargar: {len(missing_ids):,}")

    OUT_ERRORS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ERRORS, "w", encoding="utf-8") as f:
        for vid in sorted(missing_ids):
            f.write(vid + "\n")
    print(f"📝 Todos los faltantes guardados en: {OUT_ERRORS}")

    # -----------------------------
    # Separar faltantes por ciudad
    # -----------------------------
    print("\n📊 Dividiendo faltantes por ciudad...")

    for city_name, qid in CITY_QIDS.items():
        subset = df[df["venue_city"] == qid]
        ids_city = set(subset["venue_id"].astype(str))
        missing_city = sorted([vid for vid in ids_city if vid in missing_ids])

        out_path = Path(f"data/erros/MISSING_POIS_{city_name.capitalize()}.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for vid in missing_city:
                f.write(vid + "\n")

        print(f"🏙️  {city_name.capitalize()}: {len(missing_city):,} faltantes -> {out_path}")

    print("\n✅ Comprobación y división completadas.")

if __name__ == "__main__":
    main()
