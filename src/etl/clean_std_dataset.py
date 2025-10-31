import pandas as pd
from pathlib import Path

# Rutas
STD_PATH = Path("data/raw/std_2018.csv")
OUT_PATH = Path("data/processed/std_2018_clean.csv")

# Ciudades válidas (tal como vienen en el RAW, con prefijo 'wd:')
VALID_CITIES_RAW = ["wd:Q35765", "wd:Q406", "wd:Q864965"]  # Osaka, Estambul, Petaling Jaya

def remove_prefix(value: str, prefix: str) -> str:
    if isinstance(value, str) and value.startswith(prefix):
        return value.replace(prefix, "", 1)
    return value

def normalize_field(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = remove_prefix(value, "foursquare:")
    value = remove_prefix(value, "schema:")
    value = remove_prefix(value, "wd:")
    return value.strip()

def reindex_series_to_start_at_1(series: pd.Series) -> pd.Series:
    """Mapea valores únicos de la serie a 1..N preservando el orden de aparición."""
    uniques = pd.Index(series.dropna().unique())
    mapping = {old: i+1 for i, old in enumerate(uniques)}
    return series.map(mapping)

def main():
    print("📥 Leyendo dataset original...")
    df = pd.read_csv(STD_PATH)
    print(f"Registros originales: {len(df):,}")

    # 1) Filtrado por ciudades (usando los QIDs con 'wd:' del RAW)
    df = df[df["venue_city"].isin(VALID_CITIES_RAW)].copy()
    print(f"✔ Registros tras filtrar ciudades: {len(df):,}")

    # 2) Drop nulos básicos y limpieza de espacios
    df = df.dropna(subset=["venue_id", "timestamp"])
    df["venue_id"] = df["venue_id"].astype(str).str.strip()
    df["venue_category"] = df["venue_category"].astype(str).str.strip()
    df["venue_schema"] = df["venue_schema"].astype(str).str.strip()
    df["venue_city"] = df["venue_city"].astype(str).str.strip()
    df["venue_country"] = df["venue_country"].astype(str).str.strip()

    # 3) Normalización de prefijos
    cols_to_norm = ["venue_id", "venue_category", "venue_schema", "venue_city", "venue_country"]
    for c in cols_to_norm:
        if c in df.columns:
            df[c] = df[c].map(normalize_field)

    # 4) Normalización de la fecha → UTC (mantener en la misma columna 'timestamp')
    #    pandas reconoce offsets como +03:00; utc=True la convierte a UTC.
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # Quita filas con timestamp inválido si hubiera
    bad = ts.isna().sum()
    if bad:
        print(f"⚠️  Eliminando {bad} filas con timestamp inválido.")
    df = df.loc[~ts.isna()].copy()
    df["timestamp"] = ts.loc[~ts.isna()].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 5) Renumerar IDs (guardando los originales)
    if "trail_id" in df.columns:
        df["trail_id_orig"] = df["trail_id"]
        df["trail_id"] = reindex_series_to_start_at_1(df["trail_id_orig"]).astype(int)
    if "user_id" in df.columns:
        df["user_id_orig"] = df["user_id"]
        df["user_id"] = reindex_series_to_start_at_1(df["user_id_orig"]).astype(int)

    # 6) Guardar
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Guardado: {OUT_PATH} ({len(df):,} filas)")

    # 7) Resúmenes rápidos
    print("\n📊 Distribución por ciudad (QIDs sin 'wd:'):")
    print(df["venue_city"].value_counts())

    # Cobertura de renumeración
    n_users = df["user_id"].nunique() if "user_id" in df.columns else 0
    n_trails = df["trail_id"].nunique() if "trail_id" in df.columns else 0
    print(f"\n👤 Usuarios únicos (renumerados): {n_users}")
    print(f"🧭 Trails únicos (renumerados):   {n_trails}")

if __name__ == "__main__":
    main()
