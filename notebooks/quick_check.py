"""
quick_check.py: diagnósticos rápidos sobre std_clean y POIs enriquecidos.

Uso:
  python notebooks/quick_check.py

Genera reportes en data/reports/diagnostics y muestra resúmenes por consola.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# Rutas principales
STD_CLEAN = Path("data/processed/std_clean.csv")
POIS_BY_CITY = {
    "istanbul": Path("data/processed/pois_enriched_Istanbul.json"),
    "osaka": Path("data/processed/pois_enriched_Osaka.json"),
    "petalingjaya": Path("data/processed/pois_enriched_PetalingJaya.json"),
}
# QIDs tal como están en std_clean (sin prefijo 'wd:')
CITY_QIDS = {"istanbul": "Q406", "osaka": "Q35765", "petalingjaya": "Q864965"}

REPORT_DIR = Path("data/reports/diagnostics")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


def summarize_std(std_path: Path):
    if not std_path.exists():
        raise SystemExit(f"No existe {std_path}")
    df = pd.read_csv(std_path)
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    summary = {
        "rows": len(df),
        "users": df["user_id"].nunique() if "user_id" in df.columns else None,
        "trails": df["trail_id"].nunique() if "trail_id" in df.columns else None,
        "venues": df["venue_id"].nunique(),
        "cities_qid_present": df["venue_city"].value_counts().to_dict(),
        "ts_min_utc": str(ts.min()) if len(ts) else None,
        "ts_max_utc": str(ts.max()) if len(ts) else None,
    }
    df["venue_city"].value_counts().to_csv(REPORT_DIR / "std_city_distribution.csv")
    if "user_id" in df.columns:
        df["user_id"].value_counts().to_csv(REPORT_DIR / "std_users_counts.csv")
    if "trail_id" in df.columns:
        df["trail_id"].value_counts().to_csv(REPORT_DIR / "std_trails_counts.csv")

    print("=== STD_CLEAN SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>18}: {v}")
    print()
    return df


def summarize_pois():
    per_city = {}
    all_ids = set()
    (REPORT_DIR / "per_city").mkdir(parents=True, exist_ok=True)

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
        pd.Series(dict(cat_counter.most_common())).to_csv(
            REPORT_DIR / "per_city" / f"top_categories_{city}.csv"
        )
        pd.Series(sorted(set(ids))).to_csv(
            REPORT_DIR / "per_city" / f"fsq_ids_{city}.csv", index=False
        )

    print("=== POIS SUMMARY BY CITY ===")
    for city, info in per_city.items():
        print(f"\n[{city}] file={info['file']}")
        for k in [
            "pois_in_file",
            "unique_fsq_ids",
            "duplicate_ids",
            "null_city",
            "null_rating",
            "null_price",
            "null_total_ratings",
            "invalid_latlon",
        ]:
            print(f"  {k:>20}: {info[k]}")
        print("  top_categories:", info["top_categories"][:5])
    print()
    return per_city, all_ids


def coverage_std_vs_pois(std_df):
    per_city_coverage = {}
    missing_by_city = {}
    (REPORT_DIR / "missing").mkdir(parents=True, exist_ok=True)

    for city, qid in CITY_QIDS.items():
        sub = std_df[std_df["venue_city"] == qid]
        ids_std_city = set(sub["venue_id"].astype(str))

        ids_poi = set()
        poi_ids_path = REPORT_DIR / "per_city" / f"fsq_ids_{city}.csv"
        if poi_ids_path.exists():
            s = pd.read_csv(poi_ids_path, header=None).iloc[:, 0].astype(str)
            ids_poi = set(s.tolist())

        matched = ids_std_city & ids_poi
        missing = ids_std_city - ids_poi

        per_city_coverage[city] = {
            "std_ids": len(ids_std_city),
            "poi_ids": len(ids_poi),
            "matched": len(matched),
            "coverage_pct": pct(len(matched), len(ids_std_city)),
        }
        missing_by_city[city] = sorted(missing)

        with (REPORT_DIR / "missing" / f"missing_ids_{city}.txt").open(
            "w", encoding="utf-8"
        ) as f:
            for vid in missing_by_city[city]:
                f.write(vid + "\n")

    # Global
    ids_std_all = set(std_df["venue_id"].astype(str))
    ids_poi_all = set()
    for city in CITY_QIDS.keys():
        p = REPORT_DIR / "per_city" / f"fsq_ids_{city}.csv"
        if p.exists():
            s = pd.read_csv(p, header=None).iloc[:, 0].astype(str)
            ids_poi_all |= set(s.tolist())

    matched_all = ids_std_all & ids_poi_all
    global_cov = {
        "std_ids_total": len(ids_std_all),
        "poi_ids_total": len(ids_poi_all),
        "matched_total": len(matched_all),
        "coverage_pct": pct(len(matched_all), len(ids_std_all)),
    }

    print("=== COVERAGE STD vs POIs (enriched) ===")
    for city, row in per_city_coverage.items():
        print(
            f"{city:>14} | std={row['std_ids']:6d} poi={row['poi_ids']:6d} "
            f"matched={row['matched']:6d} -> {row['coverage_pct']:5.1f}%"
        )
    print("-" * 60)
    print(
        f"{'GLOBAL':>14} | std={global_cov['std_ids_total']:6d} "
        f"poi={global_cov['poi_ids_total']:6d} matched={global_cov['matched_total']:6d} "
        f"-> {global_cov['coverage_pct']:5.1f}%\n"
    )
    return per_city_coverage, global_cov


def catalog_categories():
    per_city_rows = {}
    global_counter = defaultdict(int)  # (cat_id, name) -> count
    id_to_names = defaultdict(set)  # cat_id -> set(names vistos)

    for city, path in POIS_BY_CITY.items():
        data = load_json_list(path)
        local_counter = defaultdict(int)
        for obj in data:
            cats = obj.get("categories") or []
            for c in cats:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("id")) if c.get("id") is not None else None
                cname = c.get("name")
                if not cid or not cname:
                    continue
                local_counter[(cid, cname)] += 1
                global_counter[(cid, cname)] += 1
                id_to_names[cid].add(cname)

        rows = [(cid, cname, cnt) for (cid, cname), cnt in local_counter.items()]
        per_city_rows[city] = rows
        pd.DataFrame(rows, columns=["category_id", "name", "count"]).to_csv(
            REPORT_DIR / "per_city" / f"categories_{city}.csv", index=False
        )

    # Global
    global_rows = [(cid, cname, cnt) for (cid, cname), cnt in global_counter.items()]
    pd.DataFrame(global_rows, columns=["category_id", "name", "count"]).to_csv(
        REPORT_DIR / "categories_global.csv", index=False
    )
    # Inconsistencias id->name
    inconsistencies = {
        cid: sorted(list(names)) for cid, names in id_to_names.items() if len(names) > 1
    }
    if inconsistencies:
        with (REPORT_DIR / "categories_inconsistencies.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(inconsistencies, f, ensure_ascii=False, indent=2)

    print("=== CATEGORY CATALOG ===")
    print(f"Global categories: {len(global_counter)}")
    print(f"Inconsistencies id->name: {len(inconsistencies)} (ver categories_inconsistencies.json)")


def main():
    std_df = summarize_std(STD_CLEAN)
    summarize_pois()
    coverage_std_vs_pois(std_df)
    catalog_categories()


if __name__ == "__main__":
    main()
