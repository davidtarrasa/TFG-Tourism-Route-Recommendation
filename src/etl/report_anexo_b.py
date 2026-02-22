"""
Anexo B reporting helper (ETL/data quality + simple real-route visualization).

Usage:
  python -m src.etl.report_anexo_b --dsn "postgresql://user:pass@host:port/db"
  python -m src.etl.report_anexo_b
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

try:
    import folium
except Exception:  # pragma: no cover
    folium = None


CITY_SPECS = [
    {"qid": "Q35765", "name": "Osaka", "slug": "osaka"},
    {"qid": "Q406", "name": "Istanbul", "slug": "istanbul"},
    {"qid": "Q864965", "name": "Petaling Jaya", "slug": "petalingjaya"},
]


@dataclass
class DataBundle:
    source: str  # "db" or "files"
    visits: pd.DataFrame
    pois: pd.DataFrame
    poi_categories: pd.DataFrame


def _clean_venue_id(v: object) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.startswith("foursquare:"):
        s = s.replace("foursquare:", "", 1)
    return s


def _clean_city_qid(v: object) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.startswith("wd:"):
        s = s.replace("wd:", "", 1)
    return s


def _load_from_db(dsn: str) -> DataBundle:
    import psycopg

    with psycopg.connect(dsn) as conn:
        visits = pd.read_sql("SELECT * FROM visits", conn)
        pois = pd.read_sql("SELECT * FROM pois", conn)
        poi_categories = pd.read_sql("SELECT * FROM poi_categories", conn)

    return DataBundle(source="db", visits=visits, pois=pois, poi_categories=poi_categories)


def _load_json_list(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _load_pois_files() -> pd.DataFrame:
    all_rows: List[dict] = []
    for c in CITY_SPECS:
        qid = c["qid"]
        name = c["name"]
        candidates = [
            Path(f"data/processed/foursquare/ALL_POIS_{name.replace(' ', '')}_prof_filtered.json"),
            Path(f"data/processed/pois_enriched_{name.replace(' ', '')}.json"),
            Path(f"data/raw/ALL_POIS_{name.replace(' ', '')}.json"),
        ]
        rows: List[dict] = []
        for p in candidates:
            rows = _load_json_list(p)
            if rows:
                break
        for r in rows:
            fsq_id = r.get("fsq_id") or r.get("fsq_place_id")
            if not fsq_id:
                continue
            all_rows.append(
                {
                    "fsq_id": str(fsq_id),
                    "name": r.get("name"),
                    "lat": r.get("lat") if r.get("lat") is not None else r.get("latitude"),
                    "lon": r.get("lon") if r.get("lon") is not None else r.get("longitude"),
                    "city": r.get("city"),
                    "city_qid": qid,
                    "rating": r.get("rating"),
                    "price_tier": r.get("price_tier") if r.get("price_tier") is not None else r.get("price"),
                    "total_ratings": r.get("total_ratings"),
                    "primary_category": r.get("primary_category"),
                    "is_free": r.get("is_free", False),
                    "_categories_raw": r.get("categories"),
                }
            )
    if not all_rows:
        return pd.DataFrame(
            columns=[
                "fsq_id",
                "name",
                "lat",
                "lon",
                "city",
                "city_qid",
                "rating",
                "price_tier",
                "total_ratings",
                "primary_category",
                "is_free",
                "_categories_raw",
            ]
        )
    return pd.DataFrame(all_rows).drop_duplicates(subset=["fsq_id"], keep="first")


def _build_poi_categories_from_pois(pois: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, r in pois.iterrows():
        fid = str(r.get("fsq_id"))
        cats = r.get("_categories_raw")
        if isinstance(cats, list):
            for idx, c in enumerate(cats):
                if not isinstance(c, dict):
                    continue
                cid = c.get("id") or c.get("fsq_category_id") or c.get("category_id")
                cname = c.get("name") or c.get("category_name")
                if cid and cname:
                    rows.append(
                        {
                            "fsq_id": fid,
                            "category_id": str(cid),
                            "category_name": str(cname),
                            "ord": idx,
                        }
                    )
    return pd.DataFrame(rows, columns=["fsq_id", "category_id", "category_name", "ord"])


def _load_from_files() -> DataBundle:
    visits_path = Path("data/processed/std_clean.csv")
    if not visits_path.exists():
        visits_path = Path("data/processed/std_2018_clean.csv")
    if not visits_path.exists():
        visits_path = Path("data/raw/std_2018.csv")
    visits = pd.read_csv(visits_path)

    pois = _load_pois_files()
    poi_categories = _build_poi_categories_from_pois(pois)
    return DataBundle(source="files", visits=visits, pois=pois, poi_categories=poi_categories)


def _prepare(bundle: DataBundle) -> DataBundle:
    visits = bundle.visits.copy()
    pois = bundle.pois.copy()
    poi_categories = bundle.poi_categories.copy()

    if "venue_id" in visits.columns:
        visits["venue_id"] = visits["venue_id"].map(_clean_venue_id)
    if "venue_city" in visits.columns:
        visits["venue_city"] = visits["venue_city"].map(_clean_city_qid)
    if "timestamp" in visits.columns:
        visits["timestamp"] = pd.to_datetime(visits["timestamp"], errors="coerce", utc=True)

    if "fsq_id" in pois.columns:
        pois["fsq_id"] = pois["fsq_id"].astype(str)
    if "city_qid" in pois.columns:
        pois["city_qid"] = pois["city_qid"].map(_clean_city_qid)

    for col in ("lat", "lon", "rating", "price_tier", "total_ratings"):
        if col in pois.columns:
            pois[col] = pd.to_numeric(pois[col], errors="coerce")

    return DataBundle(source=bundle.source, visits=visits, pois=pois, poi_categories=poi_categories)


def _city_pois(pois: pd.DataFrame, city_qid: str, city_name: str) -> pd.DataFrame:
    if "city_qid" in pois.columns and pois["city_qid"].notna().any():
        sub = pois[pois["city_qid"].astype(str) == city_qid].copy()
        if not sub.empty:
            return sub
    if "city" in pois.columns:
        return pois[pois["city"].astype(str).str.lower().str.contains(city_name.lower(), na=False)].copy()
    return pois.iloc[0:0].copy()


def _top5_categories(city_pois: pd.DataFrame, poi_categories: pd.DataFrame) -> List[Tuple[str, int]]:
    if city_pois.empty:
        return []
    fsq_ids = set(city_pois["fsq_id"].astype(str))

    # Prefer explicit ord=0 if present.
    if not poi_categories.empty and {"fsq_id", "category_name"}.issubset(poi_categories.columns):
        pc = poi_categories.copy()
        pc["fsq_id"] = pc["fsq_id"].astype(str)
        pc = pc[pc["fsq_id"].isin(fsq_ids)]
        if "ord" in pc.columns:
            pc = pc[pd.to_numeric(pc["ord"], errors="coerce").fillna(999) == 0]
        if not pc.empty:
            s = pc["category_name"].value_counts().head(5)
            return list(zip(s.index.tolist(), s.values.tolist()))

    if "primary_category" in city_pois.columns:
        s = city_pois["primary_category"].fillna("Unknown").astype(str).value_counts().head(5)
        return list(zip(s.index.tolist(), s.values.tolist()))
    return []


def _compute_city_summary(bundle: DataBundle) -> pd.DataFrame:
    rows: List[dict] = []
    visits = bundle.visits
    pois = bundle.pois
    poi_categories = bundle.poi_categories

    for c in CITY_SPECS:
        qid, city_name = c["qid"], c["name"]
        v = visits[visits["venue_city"].astype(str) == qid].copy() if "venue_city" in visits.columns else visits.iloc[0:0].copy()
        p = _city_pois(pois, qid, city_name)

        n_visits = int(len(v))
        n_users = int(v["user_id"].nunique()) if "user_id" in v.columns else 0
        n_trails = int(v["trail_id"].nunique()) if "trail_id" in v.columns else 0
        n_unique_visit_pois = int(v["venue_id"].dropna().astype(str).nunique()) if "venue_id" in v.columns else 0

        visit_ids = set(v["venue_id"].dropna().astype(str)) if "venue_id" in v.columns else set()
        city_pois_ids = set(p["fsq_id"].dropna().astype(str)) if "fsq_id" in p.columns else set()
        if not city_pois_ids and "fsq_id" in pois.columns:
            city_pois_ids = set(pois["fsq_id"].dropna().astype(str))
        matched = len(visit_ids & city_pois_ids)
        coverage_pct = (100.0 * matched / n_unique_visit_pois) if n_unique_visit_pois else 0.0

        # Null quality in POIs.
        city_name_col = "city_name" if "city_name" in p.columns else ("city" if "city" in p.columns else None)
        price_col = "price" if "price" in p.columns else ("price_tier" if "price_tier" in p.columns else None)
        denom = max(len(p), 1)

        def _null_pct(col: Optional[str]) -> float:
            if col is None or col not in p.columns:
                return 100.0
            return float(100.0 * p[col].isna().sum() / denom)

        top5 = _top5_categories(p, poi_categories)
        top5_str = "; ".join([f"{k} ({v})" for k, v in top5]) if top5 else "-"

        rows.append(
            {
                "city": city_name,
                "city_qid": qid,
                "n_visits": n_visits,
                "n_users": n_users,
                "n_trails": n_trails,
                "n_unique_visit_pois": n_unique_visit_pois,
                "poi_coverage_pct": round(coverage_pct, 2),
                "poi_null_lat_pct": round(_null_pct("lat"), 2),
                "poi_null_lon_pct": round(_null_pct("lon"), 2),
                "poi_null_city_name_pct": round(_null_pct(city_name_col), 2),
                "poi_null_rating_pct": round(_null_pct("rating"), 2),
                "poi_null_price_pct": round(_null_pct(price_col), 2),
                "poi_null_total_ratings_pct": round(_null_pct("total_ratings"), 2),
                "top5_primary_categories": top5_str,
            }
        )
    return pd.DataFrame(rows)


def _plot_etl_summary(df: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = df["city"].tolist()
    y = df["poi_coverage_pct"].tolist()
    bars = ax.bar(x, y, color=["#3b82f6", "#14b8a6", "#f59e0b"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cobertura POIs (%)")
    ax.set_title("Cobertura de POIs por ciudad (visitas con match en POIs)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_md_summary(df: pd.DataFrame, out_md: Path, source: str) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# ETL city summary (Anexo B)\n\n")
        f.write(f"- Fuente de datos: **{source}**\n")
        f.write("- Ciudades: Osaka, Istanbul, Petaling Jaya\n\n")
        cols = [
            "city",
            "n_visits",
            "n_users",
            "n_trails",
            "n_unique_visit_pois",
            "poi_coverage_pct",
            "poi_null_lat_pct",
            "poi_null_lon_pct",
            "poi_null_city_name_pct",
            "poi_null_rating_pct",
            "poi_null_price_pct",
            "poi_null_total_ratings_pct",
        ]
        f.write("## Tabla resumen\n\n")
        f.write(df[cols].to_string(index=False))
        f.write("\n\n## Top 5 categorías principales por ciudad\n\n")
        for _, r in df.iterrows():
            f.write(f"- **{r['city']}**: {r['top5_primary_categories']}\n")


def _select_real_trail(visits: pd.DataFrame, pois_city: pd.DataFrame, city_qid: str) -> pd.DataFrame:
    if visits.empty or pois_city.empty:
        return pd.DataFrame()
    v = visits[visits["venue_city"].astype(str) == city_qid].copy()
    if v.empty:
        return pd.DataFrame()
    p = pois_city[["fsq_id", "name", "lat", "lon"]].copy()
    p["fsq_id"] = p["fsq_id"].astype(str)
    m = v.merge(p, left_on="venue_id", right_on="fsq_id", how="inner")
    m = m.dropna(subset=["lat", "lon", "timestamp"])
    if m.empty:
        return pd.DataFrame()
    m = m.sort_values(["trail_id", "timestamp"])
    lengths = m.groupby("trail_id")["venue_id"].count().sort_values(ascending=False)
    lengths = lengths[lengths >= 6]
    if lengths.empty:
        return pd.DataFrame()
    trail_id = lengths.index[0]
    trail = m[m["trail_id"] == trail_id].copy()
    trail["step"] = range(1, len(trail) + 1)
    return trail


def _save_route_map(trail: pd.DataFrame, city_slug: str, out_html: Path) -> bool:
    if folium is None or trail.empty:
        return False
    out_html.parent.mkdir(parents=True, exist_ok=True)
    center = [float(trail.iloc[0]["lat"]), float(trail.iloc[0]["lon"])]
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    coords: List[List[float]] = []
    for _, r in trail.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        coords.append([lat, lon])
        label = f"{int(r['step'])}"
        popup = f"{label}. {r.get('name', 'POI')}<br>trail_id={r.get('trail_id')}"
        folium.Marker(
            [lat, lon],
            popup=popup,
            tooltip=popup,
            icon=folium.DivIcon(
                html=(
                    "<div style='background:#2563eb;color:white;border-radius:50%;"
                    "width:22px;height:22px;line-height:22px;text-align:center;"
                    "font-size:12px;font-weight:700;border:1px solid #1e40af;'>"
                    f"{label}</div>"
                )
            ),
        ).add_to(m)

    folium.PolyLine(coords, color="#1d4ed8", weight=3, opacity=0.85).add_to(m)
    m.save(str(out_html))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Anexo B report: ETL stats + simple real-route map")
    parser.add_argument("--dsn", help="Postgres DSN. If omitted, uses PG_DSN/POSTGRES_DSN or file fallback.")
    args = parser.parse_args()

    dsn = args.dsn or os.getenv("PG_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        env_path = Path(".env")
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k in ("PG_DSN", "POSTGRES_DSN") and v:
                        dsn = v
                        break
            except Exception:
                pass

    bundle: Optional[DataBundle] = None
    if dsn:
        try:
            bundle = _load_from_db(dsn)
        except Exception:
            bundle = None
    if bundle is None:
        bundle = _load_from_files()

    bundle = _prepare(bundle)

    # Outputs
    reports_dir = Path("data/reports")
    figures_dir = reports_dir / "figures"
    maps_dir = reports_dir / "maps"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _compute_city_summary(bundle)
    csv_path = reports_dir / "etl_city_summary.csv"
    md_path = reports_dir / "etl_city_summary.md"
    png_path = figures_dir / "etl_summary.png"

    summary_df.to_csv(csv_path, index=False, encoding="utf-8")
    _write_md_summary(summary_df, md_path, source=bundle.source)
    _plot_etl_summary(summary_df, png_path)

    route_outputs: List[str] = []
    for c in CITY_SPECS:
        p_city = _city_pois(bundle.pois, c["qid"], c["name"])
        trail = _select_real_trail(bundle.visits, p_city, c["qid"])
        out_html = maps_dir / f"route_example_{c['slug']}.html"
        ok = _save_route_map(trail, c["slug"], out_html)
        if ok:
            route_outputs.append(str(out_html))

    # Console summary
    print("\n=== ANEXO B REPORT ===")
    print(f"Data source: {bundle.source}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary MD : {md_path}")
    print(f"Figure PNG : {png_path}")
    if route_outputs:
        print("Route maps:")
        for p in route_outputs:
            print(f" - {p}")
    else:
        print("Route maps: no valid trails with >=6 POIs + coords found.")
    print("\nCity snapshot:")
    print(
        summary_df[
            [
                "city",
                "n_visits",
                "n_users",
                "n_trails",
                "n_unique_visit_pois",
                "poi_coverage_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
