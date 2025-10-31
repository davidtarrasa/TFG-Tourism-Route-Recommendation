import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# -----------------------------
# Rutas principales (ajústalas si cambian)
# -----------------------------
STD_CLEAN = Path("data/processed/std_2018_clean.csv")

POIS_BY_CITY = {
    "istanbul": Path("data/processed/foursquare/ALL_POIS_Istanbul_prof_filtered.json"),
    "osaka": Path("data/processed/foursquare/ALL_POIS_Osaka_prof_filtered.json"),
    "petalingjaya": Path("data/processed/foursquare/ALL_POIS_PetalingJaya_prof_filtered.json"),
}

# QIDs de ciudad tal como dejaste en std_2018_clean (sin prefijo 'wd:')
CITY_QIDS = {
    "istanbul": "Q406",
    "osaka": "Q35765",
    "petalingjaya": "Q864965",
}

# Salidas opcionales
REPORT_DIR = Path("data/reports/diagnostics")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utilidades
# -----------------------------
def load_json_list(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b


def is_valid_latlon(lat, lon):
    try:
        return (-90 <= float(lat) <= 90) and (-180 <= float(lon) <= 180)
    except Exception:
        return False


def first_category_name(cat_list):
    if isinstance(cat_list, list) and cat_list:
        c = cat_list[0]
        if isinstance(c, dict):
            return c.get("name")
    return None


# -----------------------------
# 1) Resumen de std_2018_clean
# -----------------------------
def summarize_std(std_path: Path):
    if not std_path.exists():
        raise SystemExit(f"❌ No existe {std_path}")

    df = pd.read_csv(std_path)

    # Asegurar tipos mínimos
    for col in ["venue_id", "venue_city", "timestamp"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Rango temporal en UTC (ya normalizaste a Z)
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    ts_valid = ts.dropna()
    date_min = ts_valid.min()
    date_max = ts_valid.max()

    summary = {
        "rows": len(df),
        "users": df["user_id"].nunique() if "user_id" in df.columns else None,
        "trails": df["trail_id"].nunique() if "trail_id" in df.columns else None,
        "venues": df["venue_id"].nunique(),
        "cities_qid_present": df["venue_city"].value_counts().to_dict(),
        "ts_min_utc": str(date_min) if pd.notna(date_min) else None,
        "ts_max_utc": str(date_max) if pd.notna(date_max) else None,
    }

    # Guardar distribución por ciudad
    df["venue_city"].value_counts().to_csv(REPORT_DIR / "std_city_distribution.csv")

    # Guardar distribución por usuario y trail (rápido)
    if "user_id" in df.columns:
        df["user_id"].value_counts().to_csv(REPORT_DIR / "std_users_counts.csv")
    if "trail_id" in df.columns:
        df["trail_id"].value_counts().to_csv(REPORT_DIR / "std_trails_counts.csv")

    print("=== STD_2018_CLEAN SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>18}: {v}")
    print()

    return df


# -----------------------------
# 2) Resumen de JSONs del profe por ciudad
# -----------------------------
def summarize_prof_jsons():
    per_city = {}
    all_ids = set()

    for city, path in POIS_BY_CITY.items():
        data = load_json_list(path)
        n = len(data)
        ids = []
        null_city = null_rating = null_price = null_tr = 0
        invalid_coords = 0
        cat_counter = Counter()

        for obj in data:
            fid = str(obj.get("fsq_id") or "").strip()
            if fid:
                ids.append(fid)
                all_ids.add(fid)

            lat = obj.get("lat")
            lon = obj.get("lon")
            if not is_valid_latlon(lat, lon):
                invalid_coords += 1

            if obj.get("city") in (None, "", []):
                null_city += 1
            if obj.get("rating") in (None, "", []):
                null_rating += 1
            if obj.get("price") in (None, "", []):
                null_price += 1
            if obj.get("total_ratings") in (None, "", []):
                null_tr += 1

            cname = first_category_name(obj.get("categories"))
            if cname:
                cat_counter[cname] += 1

        dup_count = len(ids) - len(set(ids))

        per_city[city] = {
            "file": str(path),
            "pois_in_file": n,
            "unique_fsq_ids": len(set(ids)),
            "duplicate_ids": dup_count,
            "null_city": null_city,
            "null_rating": null_rating,
            "null_price": null_price,
            "null_total_ratings": null_tr,
            "invalid_latlon": invalid_coords,
            "top_categories": cat_counter.most_common(15),
        }

        # Guardar top categorías y listado de ids por ciudad
        (REPORT_DIR / "per_city").mkdir(exist_ok=True, parents=True)
        pd.Series(dict(cat_counter.most_common())).to_csv(
            REPORT_DIR / "per_city" / f"top_categories_{city}.csv"
        )
        pd.Series(sorted(set(ids))).to_csv(
            REPORT_DIR / "per_city" / f"fsq_ids_{city}.csv", index=False
        )

    print("=== PROF JSONS SUMMARY BY CITY ===")
    for city, info in per_city.items():
        print(f"\n[{city}] file={info['file']}")
        for k in [
            "pois_in_file","unique_fsq_ids","duplicate_ids",
            "null_city","null_rating","null_price","null_total_ratings","invalid_latlon"
        ]:
            print(f"  {k:>20}: {info[k]}")
        print("  top_categories:", info["top_categories"][:5])

    print()
    return per_city, all_ids


# -----------------------------
# 3) Cobertura std vs prof (por ciudad y global)
# -----------------------------
def coverage_std_vs_prof(std_df, prof_ids_by_city):
    # Por ciudad
    per_city_coverage = {}
    missing_by_city = {}
    for city, qid in CITY_QIDS.items():
        sub = std_df[std_df["venue_city"] == qid]
        ids_std_city = set(sub["venue_id"].astype(str))

        # ids del profe para esa ciudad
        # leemos del fichero que acabamos de cargar
        prof_ids_path = REPORT_DIR / "per_city" / f"fsq_ids_{city}.csv"
        ids_prof_city = set()
        if prof_ids_path.exists():
            s = pd.read_csv(prof_ids_path, header=None).iloc[:,0].astype(str)
            ids_prof_city = set(s.tolist())

        matched = ids_std_city & ids_prof_city
        missing = ids_std_city - ids_prof_city

        per_city_coverage[city] = {
            "std_ids": len(ids_std_city),
            "prof_ids": len(ids_prof_city),
            "matched": len(matched),
            "coverage_pct": pct(len(matched), len(ids_std_city)),
        }
        missing_by_city[city] = sorted(missing)

        # Guarda los faltantes por ciudad (para cuando tengas cuota)
        (REPORT_DIR / "missing").mkdir(exist_ok=True, parents=True)
        with (REPORT_DIR / "missing" / f"missing_ids_{city}.txt").open("w", encoding="utf-8") as f:
            for vid in missing_by_city[city]:
                f.write(vid + "\n")

    # Global
    ids_std_all = set(std_df["venue_id"].astype(str))
    ids_prof_all = set()
    for city in CITY_QIDS.keys():
        p = REPORT_DIR / "per_city" / f"fsq_ids_{city}.csv"
        if p.exists():
            s = pd.read_csv(p, header=None).iloc[:,0].astype(str)
            ids_prof_all |= set(s.tolist())

    matched_all = ids_std_all & ids_prof_all
    global_cov = {
        "std_ids_total": len(ids_std_all),
        "prof_ids_total": len(ids_prof_all),
        "matched_total": len(matched_all),
        "coverage_pct": pct(len(matched_all), len(ids_std_all)),
    }

    print("=== COVERAGE STD vs PROF ===")
    for city, row in per_city_coverage.items():
        print(f"{city:>14} | std={row['std_ids']:5d} prof={row['prof_ids']:5d} "
              f"matched={row['matched']:5d} → {row['coverage_pct']:5.1f}%")
    print(f"{'-'*60}")
    print(f"{'GLOBAL':>14} | std={global_cov['std_ids_total']:5d} "
          f"prof={global_cov['prof_ids_total']:5d} matched={global_cov['matched_total']:5d} "
          f"→ {global_cov['coverage_pct']:5.1f}%\n")

    return per_city_coverage, global_cov

# -----------------------------
# 4) Catálogo de categorías (por ciudad y global)
# -----------------------------
from collections import defaultdict

def catalog_categories():
    """
    Recorre los ALL_POIS_*_prof_filtered.json y construye:
      - Por ciudad: conteo por (category_id, name)
      - Global: conteo agregado y detección de inconsistencias id->name
    Guarda CSVs en data/reports/diagnostics/per_city/ y uno global.
    """
    per_city_rows = {}  # city -> list[(cat_id, name, count)]
    global_counter = defaultdict(int)   # (cat_id, name) -> count
    id_to_names = defaultdict(set)      # cat_id -> set(names vistos)

    for city, path in POIS_BY_CITY.items():
        data = load_json_list(path)
        local_counter = defaultdict(int)  # (cat_id, name) -> count

        for obj in data:
            cats = obj.get("categories") or []
            for c in cats:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("id")) if c.get("id") is not None else None
                cname = c.get("name")
                if not cid or not cname:
                    continue
                key = (cid, cname)
                local_counter[key] += 1
                global_counter[key] += 1
                id_to_names[cid].add(cname)

        rows = [(cid, cname, cnt) for (cid, cname), cnt in sorted(local_counter.items(), key=lambda x: (-x[1], x[0][0]))]
        per_city_rows[city] = rows

        # export por ciudad
        (REPORT_DIR / "per_city").mkdir(exist_ok=True, parents=True)
        pd.DataFrame(rows, columns=["category_id", "name", "count"]).to_csv(
            REPORT_DIR / "per_city" / f"categories_catalog_{city}.csv", index=False
        )

    # global
    global_rows = [(cid, cname, cnt) for (cid, cname), cnt in sorted(global_counter.items(), key=lambda x: (-x[1], x[0][0]))]
    pd.DataFrame(global_rows, columns=["category_id", "name", "count"]).to_csv(
        REPORT_DIR / "categories_catalog_global.csv", index=False
    )

    # reporte rápido en consola y aviso de posibles inconsistencias id->name
    print("=== CATEGORIES CATALOG (top 20 global) ===")
    for cid, cname, cnt in global_rows[:10]:
        print(f"{cid:>6} | {cname:<35} {cnt:>6}")

    conflicts = {cid: names for cid, names in id_to_names.items() if len(names) > 1}
    if conflicts:
        print("\nPosibles inconsistencias id->name detectadas (mismo id con varios names):")
        for cid, names in list(conflicts.items())[:15]:
            print(f"  {cid}: {sorted(names)}")
        if len(conflicts) > 15:
            print(f"  ... y {len(conflicts)-15} ids más")

    print(f"\nCSVs creados: per_city/categories_catalog_<city>.csv y categories_catalog_global.csv en {REPORT_DIR.resolve()}")

# -----------------------------
# 5) Variantes de 'city' por fichero (frecuencia)
# -----------------------------
from collections import Counter

def catalog_city_variants():
    """
    Recorre los ALL_POIS_*_prof_filtered.json y cuenta las variantes de 'city'.
    Exporta un CSV por ciudad con (city_value, count) y muestra top en consola.
    """
    print("\n=== CITY VARIANTS (per file) ===")
    out_dir = REPORT_DIR / "per_city"
    out_dir.mkdir(parents=True, exist_ok=True)

    for city, path in POIS_BY_CITY.items():
        data = load_json_list(path)
        vals = [ (o.get("city") or "").strip() for o in data ]
        ctr = Counter([v for v in vals if v])
        rows = sorted(ctr.items(), key=lambda x: (-x[1], x[0]))
        # guarda CSV
        pd.DataFrame(rows, columns=["city_value", "count"]).to_csv(
            out_dir / f"city_variants_{city}.csv", index=False
        )
        # imprime top 10
        print(f"[{city}] total non-empty variants: {len(ctr)}")
        for val, cnt in rows[:10]:
            print(f"  {val}  ->  {cnt}")
    print(f"🗂️  CSVs creados: per_city/city_variants_<city>.csv en {REPORT_DIR.resolve()}")

# -----------------------------
# main
# -----------------------------
def main():
    std_df = summarize_std(STD_CLEAN)
    per_city_summary, _ = summarize_prof_jsons()
    coverage_std_vs_prof(std_df, per_city_summary)

    print(f"📝 Reportes CSV/TXT generados en: {REPORT_DIR.resolve()}")
    print("Listados útiles:")
    print("  - std_city_distribution.csv")
    print("  - per_city/top_categories_<city>.csv")
    print("  - per_city/fsq_ids_<city>.csv")
    print("  - missing/missing_ids_<city>.txt")

    std_df = summarize_std(STD_CLEAN)
    per_city_summary, _ = summarize_prof_jsons()
    coverage_std_vs_prof(std_df, per_city_summary)
    catalog_categories()
    print(f"📝 Reportes CSV/TXT generados en: {REPORT_DIR.resolve()}")

    # Variantes de 'city' por fichero
    catalog_city_variants()

if __name__ == "__main__":
    main()
