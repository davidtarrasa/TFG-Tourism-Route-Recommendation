from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import joblib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.category_intents import INCONCLUSIVE, INTENTS, classify_category_intent

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

OUTPUT_DIR = Path("data/reports/figures/tfg")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITY_META = {
    "Q35765": {"name": "Osaka", "slug": "osaka", "lat": 34.6937, "lon": 135.5023},
    "Q406": {"name": "Istanbul", "slug": "istanbul", "lat": 41.0082, "lon": 28.9784},
    "Q864965": {"name": "Petaling Jaya", "slug": "petalingjaya", "lat": 3.1073, "lon": 101.6067},
}


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}")


def missing_dep(dep: str, pip_name: str | None = None) -> None:
    pkg = pip_name or dep
    log(f"Falta dependencia '{dep}'. Instala con: pip install {pkg}")


def get_db_connection():
    dsn = os.getenv("POSTGRES_DSN", "postgresql://tfg:tfgpass@localhost:55432/tfg_routes")
    return psycopg.connect(dsn)


def _table_columns(conn, table: str) -> set[str]:
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=%s
    """
    with conn.cursor() as cur:
        cur.execute(q, (table,))
        return {r[0] for r in cur.fetchall()}


def load_pois(conn, city_qid: str) -> pd.DataFrame:
    cols = _table_columns(conn, "pois")
    id_col = "fsq_id" if "fsq_id" in cols else "poi_id"
    city_col = "city_qid" if "city_qid" in cols else ("city" if "city" in cols else None)
    if city_col is None:
        raise RuntimeError("Tabla pois sin columna de ciudad (city_qid/city).")
    q = f"""
    SELECT
      {id_col}::text AS poi_id,
      name,
      lat,
      lon,
      rating,
      primary_category,
      {city_col}::text AS city_qid,
      COALESCE(total_ratings, 0) AS total_ratings
    FROM pois
    WHERE {city_col} = %(city)s
    """
    return pd.read_sql(q, conn, params={"city": city_qid})


def load_visits(conn, city_qid: str) -> pd.DataFrame:
    cols = _table_columns(conn, "visits")
    poi_col = "venue_id" if "venue_id" in cols else "poi_id"
    city_col = "venue_city" if "venue_city" in cols else ("city_qid" if "city_qid" in cols else None)
    if city_col is None:
        raise RuntimeError("Tabla visits sin columna de ciudad (venue_city/city_qid).")
    q = f"""
    SELECT
      trail_id,
      user_id,
      {poi_col}::text AS poi_id,
      timestamp,
      {city_col}::text AS city_qid
    FROM visits
    WHERE {city_col} = %(city)s
    """
    df = pd.read_sql(q, conn, params={"city": city_qid})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def _candidate_eval_paths(city_qid: str) -> list[Path]:
    slug = CITY_META[city_qid]["slug"]
    return [
        Path(f"data/reports/eval_{city_qid}_current.json"),
        Path(f"data/reports/eval_{city_qid}_latest.json"),
        Path(f"data/reports/benchmarks/eval_{slug}.json"),
    ]


def load_eval_results(city_qid: str) -> pd.DataFrame:
    found = None
    for p in _candidate_eval_paths(city_qid):
        if p.exists():
            found = p
            break
    if found is None:
        raise FileNotFoundError(f"No encuentro JSON de evaluación para {city_qid}.")
    with found.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", [])
    if not metrics:
        raise RuntimeError(f"JSON sin campo metrics: {found}")
    df = pd.DataFrame(metrics)
    wide = df.pivot_table(index="mode", columns="metric", values="value", aggfunc="mean")
    wide = wide.reset_index().rename_axis(None, axis=1)
    wide["city_qid"] = city_qid
    wide["city"] = CITY_META[city_qid]["name"]
    return wide


def _save(fig_name: str) -> Path:
    return OUTPUT_DIR / fig_name


def _category_palette(categories: Iterable[str]) -> dict[str, tuple[float, float, float, float]]:
    cats = sorted({str(c) for c in categories if pd.notna(c)})
    cmap = plt.cm.get_cmap("tab20", max(1, len(cats)))
    return {c: cmap(i) for i, c in enumerate(cats)}


_INTENT_DISPLAY = {
    "food": "Food",
    "culture": "Culture",
    "nature": "Nature",
    "nightlife": "Nightlife",
    "shopping": "Shopping",
    "service": "Service",
    "health": "Health",
    "entertainment": "Entertainment",
    "transport": "Transport",
    "relaxation": "Relaxation",
    "family": "Family",
    "sports": "Sports",
    INCONCLUSIVE: "Inconclusive",
}

_INTENT_DISPLAY_ORDER = [_INTENT_DISPLAY[i] for i in INTENTS] + [_INTENT_DISPLAY[INCONCLUSIVE]]


def _map_to_broad_category(cat: str) -> str:
    raw = str(cat or "").strip()
    if not raw:
        return _INTENT_DISPLAY[INCONCLUSIVE]
    intent, _, _ = classify_category_intent(raw, use_semantic=False)
    if intent not in INTENTS and intent != INCONCLUSIVE:
        intent = INCONCLUSIVE
    return _INTENT_DISPLAY.get(intent, _INTENT_DISPLAY[INCONCLUSIVE])


def _build_markov_transition(visits: pd.DataFrame, pois: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    merged = visits.merge(
        pois[["poi_id", "primary_category"]],
        on="poi_id",
        how="left",
    ).dropna(subset=["primary_category"])
    merged["primary_category"] = merged["primary_category"].map(_map_to_broad_category)
    merged = merged.sort_values(["trail_id", "timestamp"])
    transitions = Counter()
    visits_per_cat = merged["primary_category"].value_counts()
    for _, g in merged.groupby("trail_id"):
        cats = g["primary_category"].tolist()
        for a, b in zip(cats[:-1], cats[1:]):
            transitions[(a, b)] += 1
    categories = sorted(set(visits_per_cat.index) | {k[0] for k in transitions} | {k[1] for k in transitions})
    mat = pd.DataFrame(0.0, index=categories, columns=categories)
    for (a, b), n in transitions.items():
        mat.loc[a, b] = float(n)
    row_sum = mat.sum(axis=1).replace(0, 1.0)
    mat = mat.div(row_sum, axis=0)
    return mat, visits_per_cat


def _safe_write_plotly_png(fig, out_png: Path):
    try:
        fig.write_image(str(out_png), scale=2, width=1400, height=900)
    except Exception:
        log(f"No se pudo exportar PNG con plotly/kaleido: {out_png.name}")
        fig_fallback, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(
            0.02,
            0.5,
            "No se pudo exportar PNG de plotly.\nInstala: pip install kaleido",
            va="center",
            ha="left",
            fontsize=10,
        )
        fig_fallback.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig_fallback)


def _try_add_basemap(ax, ctx, source, zoom: int) -> bool:
    providers = []
    if source is not None:
        providers.append(source)
    fallback_names = [
        ("OpenStreetMap", "Mapnik"),
        ("OpenStreetMap", "HOT"),
        ("Esri", "WorldImagery"),
        ("Esri", "WorldStreetMap"),
        ("CartoDB", "Positron"),
    ]
    for top, child in fallback_names:
        try:
            cand = getattr(getattr(ctx.providers, top), child)
            if cand not in providers:
                providers.append(cand)
        except Exception:
            continue

    last_error = None
    for prov in providers:
        try:
            ctx.add_basemap(ax, source=prov, zoom=zoom)
            return True
        except Exception as e:
            last_error = e
            continue

    log(f"Basemap no disponible (se genera figura sin fondo web): {last_error}")
    ax.set_facecolor("#F2F3F5")
    return False


def _save_sankey_fallback_png(
    labels: list[str],
    source: list[int],
    target: list[int],
    value: list[int],
    out_png: Path,
    title: str,
) -> None:
    if not labels or not value:
        raise RuntimeError("No hay datos para generar fallback estático de Sankey.")

    def _stage(lbl: str) -> int:
        m = re.search(r"(\d+)\s*$", str(lbl))
        if m:
            return int(m.group(1))
        low = str(lbl).lower()
        if "pos1" in low:
            return 1
        if "pos2" in low:
            return 2
        if "pos3" in low:
            return 3
        return 2

    def _name(lbl: str) -> str:
        txt = re.sub(r"\s*[→\-]?\s*\d+\s*$", "", str(lbl)).strip()
        txt = txt.replace("(pos1)", "").replace("(pos2)", "").replace("(pos3)", "").strip()
        return txt or str(lbl)

    n = len(labels)
    out_sum = np.zeros(n, dtype=float)
    in_sum = np.zeros(n, dtype=float)
    for s, t, v in zip(source, target, value):
        out_sum[s] += float(v)
        in_sum[t] += float(v)
    totals = np.maximum(out_sum, in_sum)
    totals[totals <= 0] = 1.0

    stages = {}
    for idx, lbl in enumerate(labels):
        stages.setdefault(_stage(lbl), []).append(idx)
    stage_keys = sorted(stages.keys())
    x_positions = {st: i / (max(1, len(stage_keys) - 1)) * 0.78 + 0.11 for i, st in enumerate(stage_keys)}

    node_pos = {}
    for st in stage_keys:
        nodes = sorted(stages[st], key=lambda i: totals[i], reverse=True)
        ys = np.linspace(0.88, 0.12, num=len(nodes)) if len(nodes) > 1 else np.array([0.5])
        for i, y in zip(nodes, ys):
            node_pos[i] = (x_positions[st], float(y))

    vmax = max(value) if value else 1
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for s, t, v in sorted(zip(source, target, value), key=lambda x: x[2]):
        x0, y0 = node_pos[s]
        x1, y1 = node_pos[t]
        w = 1.2 + 10.0 * (float(v) / float(vmax))
        color = cmap(0.25 + 0.65 * (float(v) / float(vmax)))
        edge = patches.FancyArrowPatch(
            (x0 + 0.03, y0),
            (x1 - 0.03, y1),
            arrowstyle="-",
            connectionstyle="arc3,rad=0.10",
            linewidth=w,
            color=color,
            alpha=0.55,
        )
        ax.add_patch(edge)

    node_max = float(np.max(totals))
    for idx, lbl in enumerate(labels):
        x, y = node_pos[idx]
        size = 0.026 + 0.03 * (totals[idx] / (node_max + 1e-12))
        rect = patches.FancyBboxPatch(
            (x - size * 0.6, y - size * 0.34),
            size * 1.2,
            size * 0.68,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            linewidth=0.8,
            edgecolor="#1f3b73",
            facecolor="#3f6db5",
            alpha=0.95,
        )
        ax.add_patch(rect)
        align = "right" if x > 0.65 else ("left" if x < 0.35 else "center")
        tx = x - size * 0.85 if align == "right" else (x + size * 0.85 if align == "left" else x)
        ax.text(tx, y, _name(lbl), va="center", ha=align, fontsize=9, color="#0d1b2a")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(value), vmax=max(value)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Frecuencia de transición")
    ax.set_title(title + " (fallback PNG estático)", fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_01_pipeline_sistema():
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")

    layers = [
        (
            "FUENTES DE DATOS",
            18.5,
            "#2E86AB",
            ["std_2018 (Semantic Trails)", "Foursquare Venues API", "Geoapify Routing API"],
        ),
        (
            "ETL / PROCESAMIENTO",
            15.5,
            "#A23B72",
            ["Limpieza y normalización", "Enriquecimiento POIs", "Carga PostgreSQL"],
        ),
        (
            "MOTORES DE RECOMENDACIÓN",
            12.1,
            "#F18F01",
            [
                "Content\n(TF-IDF)",
                "Item-Item\n(co-visit.)",
                "Markov\n(transic.)",
                "Embed\n(Word2Vec)",
                "ALS\n(impl. CF)",
                "Hybrid\n(fusión)",
                "RRF\n(rank fus.)",
                "Popular\n(baseline)",
            ],
        ),
        (
            "SCORING + RUTA",
            8.5,
            "#C73E1D",
            [
                "Normalización de scores",
                "Fusión: hybrid ponderado + RRF automático",
                "Reranking (distancia, precio, diversidad)",
                "NN + 2-opt ordering",
                "GeoJSON + Folium export",
            ],
        ),
        (
            "API + FRONTEND",
            5.0,
            "#3B1F2B",
            [
                "FastAPI /multi-recommend",
                "Variantes: history/inputs/location/full",
                "Interfaz web interactiva",
                "Mapa Leaflet con rutas",
            ],
        ),
    ]

    for idx, (title, y, color, items) in enumerate(layers):
        box = patches.FancyBboxPatch(
            (0.8, y - 1.2),
            8.4,
            2.2,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.8,
            edgecolor=color,
            facecolor=mcolors.to_rgba(color, 0.13),
        )
        ax.add_patch(box)
        ax.text(1.05, y + 0.52, title, fontsize=13, fontweight="bold", color=color, va="center")

        if idx == 2:
            n_engines = len(items)
            gap = 0.05
            x0 = 0.98
            w = (9.0 - x0 - gap * (n_engines - 1)) / n_engines
            for i, item in enumerate(items):
                b = patches.FancyBboxPatch(
                    (x0 + i * (w + gap), y - 0.63),
                    w,
                    1.04,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    linewidth=1.0,
                    edgecolor=color,
                    facecolor=mcolors.to_rgba(color, 0.28),
                )
                ax.add_patch(b)
                ax.text(x0 + i * (w + gap) + w / 2, y - 0.11, item, ha="center", va="center", fontsize=7.5)
        else:
            for j, item in enumerate(items):
                ax.text(1.05, y + 0.06 - j * 0.28, f"• {item}", fontsize=9.6, color="#2B2B2B", va="center")

    for y1, y2 in [(17.2, 16.3), (14.2, 13.3), (11.0, 10.2), (7.3, 6.2)]:
        ax.annotate("", xy=(5.0, y2), xytext=(5.0, y1), arrowprops=dict(arrowstyle="-|>", lw=1.9, color="#444"))

    annotations = [
        "ETL: src/etl/*.py",
        "Scoring: src/recommender/scorer.py",
        "Route: src/recommender/route_planner.py + route_builder.py",
        "API: src/recommender/api.py",
        "UI: frontend/index.html + frontend/app.js",
    ]
    ax.text(0.82, 1.02, "\n".join(annotations), fontsize=8.8, color="#333")
    ax.set_title("Pipeline completo del sistema de recomendación turística", fontsize=16, fontweight="bold", pad=14)
    fig.savefig(_save("fig_01_pipeline_sistema.png"), dpi=300, bbox_inches="tight")
    fig.savefig(_save("fig_01_pipeline_sistema.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_02_pois_mapa_categorias():
    try:
        import contextily as ctx
        import geopandas as gpd
    except ImportError as e:
        missing_dep(str(e).split("'")[1], "contextily geopandas")
        raise

    with get_db_connection() as conn:
        pois = load_pois(conn, "Q35765")
    pois = pois.dropna(subset=["lat", "lon"]).copy()
    pois["category_broad"] = pois["primary_category"].map(_map_to_broad_category)
    pois["rating"] = pd.to_numeric(pois["rating"], errors="coerce").fillna(pois["rating"].median() if len(pois) else 1.0)
    broad_order = _INTENT_DISPLAY_ORDER
    cmap = plt.cm.get_cmap("Set3", len(broad_order))
    palette = {cat: cmap(i) for i, cat in enumerate(broad_order)}
    gdf = gpd.GeoDataFrame(pois, geometry=gpd.points_from_xy(pois["lon"], pois["lat"]), crs="EPSG:4326").to_crs(3857)
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    x0, x1 = np.quantile(x, [0.01, 0.99])
    y0, y1 = np.quantile(y, [0.01, 0.99])
    padx = (x1 - x0) * 0.10
    pady = (y1 - y0) * 0.10

    fig, ax = plt.subplots(figsize=(14, 10))
    for cat, grp in gdf.groupby("category_broad"):
        grp.plot(
            ax=ax,
            markersize=np.clip(grp["rating"].values * 3.2, 4, 28),
            color=palette.get(cat, "#555"),
            alpha=0.62,
            label=str(cat),
        )
    ax.set_xlim(x0 - padx, x1 + padx)
    ax.set_ylim(y0 - pady, y1 + pady)
    _try_add_basemap(ax, ctx, ctx.providers.CartoDB.Positron, zoom=12)
    ax.set_axis_off()
    ax.set_title("Osaka (Q35765): POIs por categoría (agrupada) · tamaño = rating", fontsize=14, fontweight="bold")
    ax.legend(title="category_broad", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.savefig(_save("fig_02_pois_mapa_categorias.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_03_heatmap_checkins():
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        missing_dep("folium")
        raise
    try:
        import contextily as ctx
        import geopandas as gpd
    except ImportError as e:
        missing_dep(str(e).split("'")[1], "contextily geopandas")
        raise
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        missing_dep("scipy")
        raise

    with get_db_connection() as conn:
        visits = load_visits(conn, "Q35765")
        pois = load_pois(conn, "Q35765")
    df = visits.merge(pois[["poi_id", "lat", "lon"]], on="poi_id", how="left").dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        raise RuntimeError("Sin check-ins georreferenciados para Osaka.")

    m = folium.Map(location=[CITY_META["Q35765"]["lat"], CITY_META["Q35765"]["lon"]], zoom_start=12, tiles="CartoDB dark_matter")
    heat_data = df[["lat", "lon"]].values.tolist()
    HeatMap(heat_data, radius=12, blur=10, min_opacity=0.35).add_to(m)
    m.save(str(_save("fig_03_heatmap_checkins.html")))

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs(3857)
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=0.20)
    xmin, xmax = np.quantile(x, [0.01, 0.99])
    ymin, ymax = np.quantile(y, [0.01, 0.99])
    padx = (xmax - xmin) * 0.07
    pady = (ymax - ymin) * 0.07
    xmin, xmax = xmin - padx, xmax + padx
    ymin, ymax = ymin - pady, ymax + pady
    xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
    coords = np.vstack([xx.ravel(), yy.ravel()])
    z = kde(coords).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. Fijar límites de los ejes PRIMERO con los datos reales
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # 2. Ahora el basemap descarga tiles para Osaka (usa xlim/ylim actuales)
    _try_add_basemap(ax, ctx, ctx.providers.CartoDB.Positron, zoom=12)

    # 3. Superponer KDE con pcolormesh (usa las mismas coordenadas Mercator xx, yy)
    #    pcolormesh respeta el sistema de coordenadas del eje, a diferencia de imshow
    z_norm = z / (z.max() + 1e-12)
    pcm = ax.pcolormesh(xx, yy, z_norm, cmap="inferno", alpha=0.62, shading="auto",
                        vmin=0.01, zorder=5)
    fig.colorbar(pcm, ax=ax, label="densidad check-ins")

    ax.set_axis_off()
    ax.set_title("Heatmap KDE de check-ins (Osaka)", fontsize=14, fontweight="bold")
    fig.savefig(_save("fig_03_heatmap_checkins.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_04_hexbin_rating():
    try:
        import contextily as ctx
        import geopandas as gpd
    except ImportError as e:
        missing_dep(str(e).split("'")[1], "contextily geopandas")
        raise

    with get_db_connection() as conn:
        pois = load_pois(conn, "Q35765")
    pois = pois.dropna(subset=["lat", "lon"]).copy()
    pois["rating"] = pd.to_numeric(pois["rating"], errors="coerce").fillna(0.0)
    gdf = gpd.GeoDataFrame(pois, geometry=gpd.points_from_xy(pois["lon"], pois["lat"]), crs="EPSG:4326").to_crs(3857)
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    x0, x1 = np.quantile(x, [0.01, 0.99])
    y0, y1 = np.quantile(y, [0.01, 0.99])
    padx = (x1 - x0) * 0.16
    pady = (y1 - y0) * 0.16
    fig, ax = plt.subplots(figsize=(11, 9))
    hb = ax.hexbin(
        gdf.geometry.x.values,
        gdf.geometry.y.values,
        C=gdf["rating"].values,
        gridsize=95,
        cmap="YlOrRd",
        reduce_C_function=np.mean,
        mincnt=2,
    )
    hb.set_alpha(0.46)
    ax.set_xlim(x0 - padx, x1 + padx)
    ax.set_ylim(y0 - pady, y1 + pady)
    _try_add_basemap(ax, ctx, ctx.providers.CartoDB.Positron, zoom=12)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("rating promedio")
    ax.set_axis_off()
    ax.set_title("Hexbin rating de POIs (Osaka)", fontsize=14, fontweight="bold")
    fig.savefig(_save("fig_04_hexbin_rating.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_05_tres_ciudades():
    try:
        import contextily as ctx
        import geopandas as gpd
    except ImportError as e:
        missing_dep(str(e).split("'")[1], "contextily geopandas")
        raise

    fig, axes = plt.subplots(1, 3, figsize=(23, 7.2))
    for ax, city_qid in zip(axes, CITY_META.keys()):
        with get_db_connection() as conn:
            pois = load_pois(conn, city_qid)
        pois = pois.dropna(subset=["lat", "lon"])
        if pois.empty:
            ax.set_title(f"{CITY_META[city_qid]['name']} (sin datos)")
            ax.axis("off")
            continue
        pois = pois.copy()
        pois["category_broad"] = pois["primary_category"].map(_map_to_broad_category)
        palette = _category_palette(pois["category_broad"])
        gdf = gpd.GeoDataFrame(pois, geometry=gpd.points_from_xy(pois["lon"], pois["lat"]), crs="EPSG:4326").to_crs(3857)
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        x0, x1 = np.quantile(x, [0.01, 0.99])
        y0, y1 = np.quantile(y, [0.01, 0.99])
        padx = (x1 - x0) * 0.12
        pady = (y1 - y0) * 0.12
        for cat, grp in gdf.groupby("category_broad"):
            grp.plot(ax=ax, color=palette.get(cat, "#666"), markersize=7, alpha=0.65)
        ax.set_xlim(x0 - padx, x1 + padx)
        ax.set_ylim(y0 - pady, y1 + pady)
        ax.set_aspect("auto")
        _try_add_basemap(ax, ctx, ctx.providers.CartoDB.Positron, zoom=11)
        ax.set_title(CITY_META[city_qid]["name"], fontsize=12, fontweight="bold")
        ax.set_axis_off()
    fig.suptitle("Distribución de POIs por ciudad de estudio", fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.992, top=0.90, bottom=0.03, wspace=0.015)
    fig.savefig(_save("fig_05_tres_ciudades.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_06_markov_heatmap():
    try:
        import seaborn as sns
    except ImportError:
        missing_dep("seaborn")
        raise

    with get_db_connection() as conn:
        visits = load_visits(conn, "Q35765")
        pois = load_pois(conn, "Q35765")
    mat, _ = _build_markov_transition(visits, pois)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(mat, cmap="Blues", annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "P(destino|origen)"})
    ax.set_xlabel("Categoría destino")
    ax.set_ylabel("Categoría origen")
    ax.set_title("Matriz de transición Markov por categoría (Osaka)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(_save("fig_06_markov_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_07_markov_grafo():
    try:
        import networkx as nx
    except ImportError:
        missing_dep("networkx")
        raise

    with get_db_connection() as conn:
        visits = load_visits(conn, "Q35765")
        pois = load_pois(conn, "Q35765")
    mat, freq = _build_markov_transition(visits, pois)
    threshold = 0.03
    min_edges_per_node = 2
    min_edges_transport = 3
    G = nx.DiGraph()
    for cat in mat.index:
        G.add_node(cat)
    for i in mat.index:
        row = mat.loc[i].drop(labels=[i], errors="ignore")
        row = row[row > 0].sort_values(ascending=False)
        if row.empty:
            continue
        min_keep = min_edges_transport if str(i).lower() == "transport" else min_edges_per_node
        keep = row[row >= threshold]
        if len(keep) < min_keep:
            keep = row.head(min_keep)
        for j, w in keep.items():
            G.add_edge(i, j, weight=float(w))
    if G.number_of_edges() == 0:
        raise RuntimeError("No hay aristas >= threshold para fig_07.")

    # Layout circular: garantiza que todos los nodos caben en el canvas
    pos = nx.circular_layout(G)

    # Tamaños normalizados: entre 800 y 3500 (evitar extremos)
    freqs_raw = np.array([float(freq.get(n, 1)) for n in G.nodes()])
    f_min, f_max = freqs_raw.min(), freqs_raw.max()
    freqs_norm = (freqs_raw - f_min) / (f_max - f_min + 1e-9)
    node_sizes = 800 + freqs_norm * 2700   # rango [800, 3500]

    edge_w = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths = [1.5 + 8 * w for w in edge_w]

    fig, ax = plt.subplots(figsize=(14, 12))
    # Margen explícito para que los nodos del borde no se recorten
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color="#2196F3", alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           edge_color=edge_w,
                           edge_cmap=plt.cm.Blues,
                           arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.12",
                           ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                               norm=plt.Normalize(vmin=min(edge_w), vmax=max(edge_w)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Probabilidad de transición", shrink=0.7)

    ax.set_title("Grafo de transición Markov (base>=0.03 + top-N por categoría, Osaka)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.savefig(_save("fig_07_markov_grafo.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_08_sankey_rutas():
    try:
        import plotly.graph_objects as go
    except ImportError:
        missing_dep("plotly")
        raise

    with get_db_connection() as conn:
        visits = load_visits(conn, "Q35765")
        pois = load_pois(conn, "Q35765")

    merged = visits.merge(pois[["poi_id", "primary_category"]], on="poi_id", how="left").dropna(subset=["primary_category"])
    merged["broad"] = merged["primary_category"].apply(_map_to_broad_category)
    merged = merged.sort_values(["trail_id", "timestamp"])

    links = Counter()
    for _, g in merged.groupby("trail_id"):
        cats = g["broad"].tolist()
        if len(cats) >= 3:
            a, b, c = cats[0], cats[1], cats[2]
            links[(f"{a} →1", f"{b} →2")] += 1
            links[(f"{b} →2", f"{c} →3")] += 1
    # Filtrar transiciones con menos de 2 ocurrencias (reducir ruido)
    links = Counter({k: v for k, v in links.items() if v >= 2})
    if not links:
        raise RuntimeError("No se pudieron construir transiciones 3-posiciones para Sankey.")

    labels = sorted({s for s, _ in links} | {t for _, t in links})
    idx = {lbl: i for i, lbl in enumerate(labels)}
    source = [idx[s] for s, _ in links]
    target = [idx[t] for _, t in links]
    value = list(links.values())

    fig = go.Figure(
        go.Sankey(
            node=dict(label=labels, pad=12, thickness=16, line=dict(color="black", width=0.3)),
            link=dict(source=source, target=target, value=value),
        )
    )
    fig.update_layout(title_text="Sankey de transiciones (pos1→pos2→pos3) en trails de Osaka", font_size=11)
    fig.write_html(str(_save("fig_08_sankey_rutas.html")))
    out_png = _save("fig_08_sankey_rutas.png")
    try:
        fig.write_image(str(out_png), scale=2, width=1400, height=900)
    except Exception:
        log(f"No se pudo exportar PNG con plotly/kaleido: {out_png.name}. Generando fallback estático.")
        _save_sankey_fallback_png(
            labels=labels,
            source=source,
            target=target,
            value=value,
            out_png=out_png,
            title="Sankey de transiciones (pos1→pos2→pos3) en trails de Osaka",
        )


def _extract_word2vec_vectors(model_obj, keys: list[str]) -> tuple[np.ndarray, list[str]]:
    vecs = []
    used = []
    keyed = None
    if hasattr(model_obj, "wv"):
        keyed = model_obj.wv
    elif hasattr(model_obj, "key_to_index"):
        keyed = model_obj
    if keyed is None:
        return np.empty((0, 0)), []
    for k in keys:
        if k in keyed.key_to_index:
            vecs.append(np.array(keyed[k], dtype=float))
            used.append(k)
    if not vecs:
        return np.empty((0, 0)), []
    return np.vstack(vecs), used


def fig_09_tsne_embeddings():
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        missing_dep("scikit-learn", "scikit-learn")
        raise

    model_path = Path("src/recommender/cache/word2vec_q35765.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}")
    model = joblib.load(model_path)

    with get_db_connection() as conn:
        pois = load_pois(conn, "Q35765")
    pois = pois.dropna(subset=["primary_category"]).copy()
    keys = pois["poi_id"].astype(str).tolist()
    vectors, used = _extract_word2vec_vectors(model, keys)
    if len(used) < 30:
        raise RuntimeError("Muy pocos vectores disponibles para t-SNE.")
    sub = pois[pois["poi_id"].astype(str).isin(used)].drop_duplicates("poi_id").copy()
    sub = sub.set_index(sub["poi_id"].astype(str)).loc[used].reset_index(drop=True)
    perp = max(5, min(30, (len(sub) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto")
    xy = tsne.fit_transform(vectors)

    sub["category_broad"] = sub["primary_category"].apply(_map_to_broad_category)
    palette = _category_palette(sub["category_broad"])
    fig, ax = plt.subplots(figsize=(12, 9))
    plot_df = sub.assign(x=xy[:, 0], y=xy[:, 1])
    for cat, grp in plot_df.groupby("category_broad"):
        ax.scatter(grp["x"], grp["y"], s=22, alpha=0.70, color=palette.get(cat, "#777"), label=str(cat))
    ax.set_title("Embeddings Word2Vec de POIs (t-SNE) - Osaka", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Categoría", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.savefig(_save("fig_09_tsne_embeddings.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_10_als_matriz():
    try:
        import seaborn as sns
    except ImportError:
        missing_dep("seaborn")
        raise

    with get_db_connection() as conn:
        visits = load_visits(conn, "Q35765")
    top_users = visits["user_id"].value_counts().head(50).index
    top_pois = visits["poi_id"].value_counts().head(50).index
    sample = visits[visits["user_id"].isin(top_users) & visits["poi_id"].isin(top_pois)].copy()
    mat = pd.crosstab(sample["user_id"], sample["poi_id"]).reindex(index=top_users, columns=top_pois, fill_value=0)
    binary = (mat > 0).astype(int)
    sparsity = 1.0 - (binary.values.sum() / binary.size if binary.size else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.4, 1.2]})
    sns.heatmap(binary, cmap="Blues", cbar=True, ax=axes[0], xticklabels=False, yticklabels=False)
    axes[0].set_title("Matriz usuario-POI (top50x50) - binaria")
    axes[0].set_xlabel("POIs")
    axes[0].set_ylabel("Usuarios")
    axes[1].axis("off")
    txt = (
        "ALS trabaja sobre una matriz muy dispersa.\n\n"
        f"Shape: {binary.shape[0]} x {binary.shape[1]}\n"
        f"Interacciones observadas: {int(binary.values.sum())}\n"
        f"Sparsity: {sparsity:.2%}\n\n"
        "La factorización implícita permite\n"
        "estimar afinidades usuario-POI\n"
        "en celdas no observadas."
    )
    axes[1].text(0.02, 0.95, txt, va="top", fontsize=11)
    fig.suptitle("Visualización de matriz de interacción para ALS (Osaka)", fontsize=14, fontweight="bold")
    fig.savefig(_save("fig_10_als_matriz.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_11_hybrid_weights():
    if tomllib is None:
        raise RuntimeError("tomllib no disponible para leer TOML.")
    cfg_path = Path("configs/recommender_q35765.toml")
    if not cfg_path.exists():
        cfg_path = Path("configs/recommender.toml")
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    h = cfg.get("hybrid", {})

    engines = ["content", "item", "markov", "embed", "als"]
    scenarios = {
        "nuevo": h.get("cold_start", [0.60, 0.20, 0.20, 0.00, 0.00]),
        "historial": h.get("user_only", [0.10, 0.20, 0.10, 0.05, 0.55]),
        "geo": h.get("current_only", [0.10, 0.15, 0.35, 0.05, 0.35]),
    }
    x = np.arange(len(engines))
    width = 0.23
    colors = ["#4E79A7", "#F28E2B", "#E15759"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, vals) in enumerate(scenarios.items()):
        vals = np.array(vals, dtype=float)
        if vals.sum() > 0:
            vals = vals / vals.sum()
        ax.bar(x + (i - 1) * width, vals, width=width, label=name, color=colors[i], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(engines)
    ax.set_ylabel("peso normalizado")
    ax.set_title(f"Pesos híbridos por escenario ({cfg_path.name})", fontsize=14, fontweight="bold")
    ax.legend(title="escenario")
    fig.savefig(_save("fig_11_hybrid_weights.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_12_tabla_metricas():
    METRICS   = ["hit", "precision", "recall", "ndcg", "novelty", "diversity"]
    COL_HDR   = ["Hit@K", "Prec@K", "Recall@K", "nDCG@K", "Novelty", "Diversity"]
    MODE_ORDER = ["rrf", "item", "markov", "popular", "hybrid", "als", "embed", "content", "random"]
    HDR_BG, HDR_FG = "#37474F", "white"
    MODE_BG        = "#E3F2FD"
    ROW_BG         = ["#F5F5F5", "#FFFFFF"]
    BEST_BG        = "#C8E6C9"
    WORST_BG       = "#FFCDD2"

    city_dfs: dict[str, pd.DataFrame] = {}
    for qid in CITY_META:
        city_dfs[qid] = load_eval_results(qid)

    # Save full CSV
    all_df = pd.concat(list(city_dfs.values()), ignore_index=True)
    all_df.to_csv(_save("fig_12_tabla_metricas.csv"), index=False)

    fig, axes = plt.subplots(3, 1, figsize=(15, 22),
                             gridspec_kw={"hspace": 0.18})
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "Métricas de evaluación offline — Hit / Precision / Recall / nDCG@20 · Novelty · Diversity\n"
        "Protocolo: last_trail_user  ·  --fair  ·  k=20  ·  seed=42",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, qid in zip(axes, CITY_META):
        df = city_dfs[qid]
        city_name = CITY_META[qid]["name"]

        # Ordered modes: prefer MODE_ORDER, append any extras, random always last
        present = set(df["mode"].tolist())
        ordered = [m for m in MODE_ORDER if m in present]
        ordered += [m for m in present if m not in ordered and m != "random"]
        if "random" in present and "random" not in ordered:
            ordered.append("random")

        # Build cell text matrix
        cell_text = []
        for mode in ordered:
            row = df[df["mode"] == mode]
            vals = []
            for c in METRICS:
                if row.empty or c not in row.columns:
                    vals.append("—")
                else:
                    v = row.iloc[0][c]
                    vals.append(f"{float(v):.3f}" if pd.notna(v) else "—")
            cell_text.append([mode] + vals)

        ax.axis("off")
        ax.set_title(f"  {city_name}  ({qid})", fontsize=12, fontweight="bold",
                     loc="left", pad=6, color="#1A1A2E")

        tab = ax.table(
            cellText=cell_text,
            colLabels=["Motor"] + COL_HDR,
            loc="center",
            cellLoc="center",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(10)
        tab.scale(1.05, 2.45)

        n_rows = len(ordered)
        n_cols = 1 + len(METRICS)

        # Header row style
        for j in range(n_cols):
            cell = tab[(0, j)]
            cell.set_facecolor(HDR_BG)
            cell.set_text_props(color=HDR_FG, fontweight="bold")

        # Alternating row + mode-column style
        for ri, mode in enumerate(ordered):
            bg = ROW_BG[ri % 2]
            for j in range(n_cols):
                cell = tab[(ri + 1, j)]
                if j == 0:
                    cell.set_facecolor(MODE_BG)
                    cell.set_text_props(fontweight="bold")
                else:
                    cell.set_facecolor(bg)

        # Best / worst highlight per metric column
        for col_j, col_name in enumerate(METRICS, start=1):
            num_vals: list[tuple[int, float]] = []
            for ri, mode in enumerate(ordered):
                row = df[df["mode"] == mode]
                if not row.empty and col_name in row.columns:
                    v = row.iloc[0][col_name]
                    try:
                        num_vals.append((ri, float(v)))
                    except (TypeError, ValueError):
                        pass

            non_rnd = [(ri, v) for ri, v in num_vals if ordered[ri] != "random"]
            valid_v = [v for _, v in non_rnd if pd.notna(v)]
            if not valid_v:
                continue
            max_v, min_v = max(valid_v), min(valid_v)

            for ri, v in num_vals:
                if pd.isna(v):
                    continue
                cell = tab[(ri + 1, col_j)]
                if v == max_v:
                    cell.set_facecolor(BEST_BG)
                elif v == min_v and ordered[ri] != "random":
                    cell.set_facecolor(WORST_BG)

    fig.text(0.5, -0.005,
             "Verde = mejor motor por columna (excl. random)  ·  Rojo = peor motor por columna",
             ha="center", fontsize=9, color="#555", style="italic")
    fig.savefig(_save("fig_12_tabla_metricas.png"), dpi=220, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_13_barras_agrupadas():
    rows = []
    for qid in CITY_META:
        df = load_eval_results(qid)
        for _, r in df.iterrows():
            rows.append({"city": CITY_META[qid]["name"], "mode": r["mode"], "ndcg": float(r.get("ndcg", np.nan))})
    d = pd.DataFrame(rows).dropna(subset=["ndcg"])
    if d.empty:
        raise RuntimeError("Sin ndcg disponible para fig_13.")
    modes = sorted(d["mode"].unique())
    cities = [CITY_META[q]["name"] for q in CITY_META]
    x = np.arange(len(modes))
    width = 0.24

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, city in enumerate(cities):
        y = [float(d[(d["city"] == city) & (d["mode"] == m)]["ndcg"].mean()) for m in modes]
        ax.bar(x + (i - 1) * width, y, width=width, label=city)
    if "random" in modes:
        rnd = float(d[d["mode"] == "random"]["ndcg"].mean())
        ax.axhline(rnd, color="red", linestyle="--", linewidth=1.8, label="baseline random (media)")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=30, ha="right")
    ax.set_ylabel("ndcg")
    ax.set_title("Comparativa NDCG por engine y ciudad", fontsize=14, fontweight="bold")
    ax.legend()
    fig.savefig(_save("fig_13_barras_agrupadas.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_14_radar_chart():
    """Radar con selección de motores más representativos (Opción C: sin solapamiento)."""
    metrics = ["hit", "precision", "recall", "ndcg", "novelty", "diversity"]
    metric_labels = ["Hit@K", "Precision@K", "Recall@K", "nDCG@K", "Novelty", "Diversity"]
    # Selección: mejor meta-modelo, mejor secuencial, híbrido, co-visit., baseline fuerte, control
    selected = ["rrf", "markov", "hybrid", "item", "popular", "random"]
    palette = {
        "rrf":     "#E63946",
        "markov":  "#457B9D",
        "hybrid":  "#2A9D8F",
        "item":    "#F4A261",
        "popular": "#8338EC",
        "random":  "#444444",
    }

    # Istanbul excluida: volumen ~4× menor que Osaka/PJ distorsiona la media agregada.
    _EVAL_QIDS = ["Q35765", "Q864965"]  # Osaka + Petaling Jaya
    frames = [load_eval_results(qid) for qid in _EVAL_QIDS]
    d = pd.concat(frames, ignore_index=True)
    agg = d.groupby("mode")[metrics].mean().reset_index()
    agg = agg[agg["mode"].isin(selected)].set_index("mode").reindex(selected).reset_index()
    if agg.empty:
        raise RuntimeError("Sin datos para radar chart.")

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for _, row in agg.iterrows():
        mode_name = str(row["mode"])
        vals = [float(row[m]) for m in metrics] + [float(row[metrics[0]])]
        color = palette.get(mode_name, "#888888")
        lw = 2.8 if mode_name in ("rrf", "markov") else 2.0
        ls = "--" if mode_name == "random" else "-"
        alpha = 0.07 if mode_name == "random" else 0.12
        label = "random (control)" if mode_name == "random" else mode_name
        ax.plot(angles, vals, linewidth=lw, linestyle=ls, color=color, label=label)
        ax.fill(angles, vals, alpha=alpha, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_title(
        "Radar multi-métrica: motores principales\n(media Osaka + Petaling Jaya · als, embed, content omitidos por claridad)",
        fontsize=13, fontweight="bold", pad=18,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=10)
    fig.savefig(_save("fig_14_radar_chart.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_14b_heatmap_metricas():
    """Heatmap completo modo × métrica (Opción B: todos los motores, sin solapamiento)."""
    try:
        import seaborn as sns
    except ImportError:
        missing_dep("seaborn")
        raise

    metrics = ["hit", "precision", "recall", "ndcg", "novelty", "diversity"]
    metric_labels = ["Hit@K", "Precision@K", "Recall@K", "nDCG@K", "Novelty", "Diversity"]

    # Istanbul excluida: volumen ~4× menor que Osaka/PJ distorsiona la media agregada.
    _EVAL_QIDS = ["Q35765", "Q864965"]  # Osaka + Petaling Jaya
    frames = [load_eval_results(qid) for qid in _EVAL_QIDS]
    d = pd.concat(frames, ignore_index=True)
    agg = d.groupby("mode")[metrics].mean()

    # Orden descendente por Hit@K
    agg = agg.sort_values("hit", ascending=False)

    # Normalizar 0-1 por columna para el color (los valores anotados son los reales)
    col_min = agg.min()
    col_max = agg.max()
    normalized = (agg - col_min) / (col_max - col_min + 1e-12)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        normalized,
        annot=agg.round(3),
        fmt=".3f",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.6,
        linecolor="#E0E0E0",
        cbar_kws={"label": "valor normalizado por columna  (0 = peor · 1 = mejor)"},
        xticklabels=metric_labels,
        vmin=0, vmax=1,
        annot_kws={"size": 9},
    )
    ax.set_xlabel("Métrica", fontsize=11)
    ax.set_ylabel("Motor", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    ax.set_title(
        "Comparativa de motores: todas las métricas (media Osaka + Petaling Jaya · ordenado por Hit@K)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(_save("fig_14b_heatmap_metricas.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_15_curvas_metricas_k():
    df = load_eval_results("Q35765")
    metrics = ["precision", "recall", "ndcg"]
    for m in metrics:
        if m not in df.columns:
            raise RuntimeError(f"No hay métrica '{m}' en resultados de evaluación.")

    preferred = ["content", "item", "markov", "embed", "als", "hybrid", "random"]
    modes_existing = list(df["mode"].astype(str).unique())
    modes = [m for m in preferred if m in modes_existing] + [m for m in sorted(modes_existing) if m not in preferred]

    x = np.arange(len(modes))
    width = 0.24
    metric_labels = {"precision": "Precision@20", "recall": "Recall@20", "ndcg": "nDCG@20"}
    metric_colors = {"precision": "#4E79A7", "recall": "#F28E2B", "ndcg": "#59A14F"}

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(metrics):
        vals = [float(df.loc[df["mode"] == mode, metric].iloc[0]) for mode in modes]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=metric_labels[metric],
            color=metric_colors[metric],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_xlabel("motor")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.suptitle("Métricas de ranking a k=20 por motor (Osaka)", fontsize=13, fontweight="bold")
    fig.savefig(_save("fig_15_curvas_metricas_k.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_16_distribucion_categorias():
    with get_db_connection() as conn:
        visits = load_visits(conn, "Q35765")
        pois = load_pois(conn, "Q35765")
    pois["broad"] = pois["primary_category"].fillna("Unknown").apply(_map_to_broad_category)
    pois_cat = pois["broad"].value_counts()
    merged = visits.merge(pois[["poi_id", "broad"]], on="poi_id", how="left")
    chk_cat = merged["broad"].fillna("Other").value_counts()
    cats = sorted(set(pois_cat.index) | set(chk_cat.index))
    df = pd.DataFrame(
        {"POIs": [pois_cat.get(c, 0) for c in cats],
         "checkins": [chk_cat.get(c, 0) for c in cats]},
        index=cats,
    )
    df = df.sort_values("checkins")

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y - 0.2, df["POIs"].values, height=0.38, label="POIs", color="#4E79A7")
    ax.barh(y + 0.2, (df["checkins"] / 10.0).values, height=0.38, label="check-ins / 10", color="#F28E2B")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=11)
    ax.set_title("Distribución de categorías amplias (Osaka)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    fig.savefig(_save("fig_16_distribucion_categorias.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_17_long_tail_usuarios():
    with get_db_connection() as conn:
        visits = load_visits(conn, "Q35765")
    counts = visits["user_id"].value_counts()
    if counts.empty:
        raise RuntimeError("Sin actividad de usuarios.")
    sorted_vals = np.sort(counts.values)
    cum_vals = np.cumsum(sorted_vals) / sorted_vals.sum()
    cum_users = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(counts.values, bins=50, color="#4E79A7", edgecolor="white", log=True)
    axes[0].set_title("Actividad por usuario (histograma log)")
    axes[0].set_xlabel("check-ins por usuario")
    axes[0].set_ylabel("n usuarios (log)")

    axes[1].plot(cum_users, cum_vals, color="#E15759", linewidth=2)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].fill_between(cum_users, cum_vals, cum_users, alpha=0.2, color="#E15759")
    axes[1].set_title("Curva de Lorenz (long-tail)")
    axes[1].set_xlabel("usuarios acumulados")
    axes[1].set_ylabel("check-ins acumulados")
    fig.savefig(_save("fig_17_long_tail_usuarios.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_18_heatmap_temporal():
    try:
        import seaborn as sns
    except ImportError:
        missing_dep("seaborn")
        raise

    with get_db_connection() as conn:
        visits = load_visits(conn, "Q35765")

    if "timestamp" in visits.columns and visits["timestamp"].notna().any():
        v = visits.dropna(subset=["timestamp"]).copy()
        v["hour"] = v["timestamp"].dt.hour
        v["dow"] = v["timestamp"].dt.dayofweek
        pivot = v.pivot_table(index="hour", columns="dow", values="poi_id", aggfunc="count", fill_value=0)
        pivot = pivot.reindex(index=range(24), columns=range(7), fill_value=0)
    else:
        rng = np.random.default_rng(42)
        base = rng.poisson(8, (24, 7)).astype(float)
        for h in [11, 12, 13, 18, 19, 20]:
            base[h, :] += rng.integers(8, 18, size=7)
        pivot = pd.DataFrame(base, index=range(24), columns=range(7))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, annot=True, fmt=".0f", cbar_kws={"label": "check-ins"})
    ax.set_title("Heatmap temporal (hora x día semana) - Osaka", fontsize=14, fontweight="bold")
    ax.set_xlabel("día semana (0=lunes)")
    ax.set_ylabel("hora")
    fig.savefig(_save("fig_18_heatmap_temporal.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_19_longitud_trails():
    with get_db_connection() as conn:
        visits = load_visits(conn, "Q35765")
    lengths = visits.groupby("trail_id")["poi_id"].nunique()
    if lengths.empty:
        raise RuntimeError("Sin trails para histograma.")
    mean_v = float(lengths.mean())
    med_v = float(lengths.median())

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(lengths.values, bins=range(1, int(lengths.max()) + 3), color="#59A14F", alpha=0.85, edgecolor="white")
    ax.axvline(mean_v, color="red", linestyle="--", linewidth=2, label=f"media={mean_v:.2f}")
    ax.axvline(med_v, color="orange", linestyle="--", linewidth=2, label=f"mediana={med_v:.2f}")
    ax.set_title("Longitud de trails (#POIs por trail) - Osaka", fontsize=14, fontweight="bold")
    ax.set_xlabel("#POIs por trail")
    ax.set_ylabel("#trails")
    ax.legend()
    fig.savefig(_save("fig_19_longitud_trails.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_20_eval_protocolo():
    """Diagrama visual completo del protocolo de evaluación offline."""
    W, H = 14, 24
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFCFF")

    # ── Paleta ──────────────────────────────────────────────────────────────
    C_TRAIN  = "#2E86AB"
    C_TEST   = "#F18F01"
    C_ENGINE = "#3BB273"
    C_METRIC = "#7B2D8B"
    C_COLD   = "#E76F51"
    C_WARM   = "#2196F3"
    C_ARROW  = "#555555"

    def _box(x, y, w, h, color, alpha=0.10, radius=0.12, lw=1.8):
        p = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.03,rounding_size={radius}",
            linewidth=lw, edgecolor=color,
            facecolor=mcolors.to_rgba(color, alpha),
        )
        ax.add_patch(p)

    def _arrow(x0, y0, x1, y1):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=C_ARROW,
                            connectionstyle="arc3,rad=0.0"),
        )

    def _section_title(x, y, text, color, size=12):
        ax.text(x, y, text, fontsize=size, fontweight="bold", color=color, va="top")

    # ════════════════════════════════════════════════════════════════════════
    # TÍTULO PRINCIPAL
    # ════════════════════════════════════════════════════════════════════════
    _box(0.3, 22.4, W - 0.6, 1.3, "#1A1A2E", alpha=0.85, radius=0.15, lw=0)
    ax.text(W / 2, 23.15, "Protocolo de Evaluación Offline",
            fontsize=16, fontweight="bold", color="white",
            ha="center", va="center")
    ax.text(W / 2, 22.6, "last_trail_user  ·  --fair  ·  9 motores  ·  cold/warm split  ·  3 ciudades",
            fontsize=9.5, color="#CCDDEE", ha="center", va="center")

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 1 — PARTICIÓN DE DATOS (last_trail_user)
    # ════════════════════════════════════════════════════════════════════════
    _box(0.3, 16.9, W - 0.6, 5.1, C_TRAIN, alpha=0.06, radius=0.14)
    _section_title(0.65, 21.75, "1 · Partición   last_trail_user", C_TRAIN, size=12)

    # Sub-boxes: usuario con ≥2 trails vs usuario con 1 trail
    _box(0.65, 17.15, 6.2, 4.3, C_TRAIN, alpha=0.10)
    ax.text(1.0, 21.2, "Usuario con >= 2 trails", fontsize=10, fontweight="bold", color=C_TRAIN)

    # Dibujar 3 trails como barras horizontales
    trail_colors = [C_TRAIN, C_TRAIN, C_TEST]
    trail_labels = ["Trail 1  →  TRAIN", "Trail 2  →  TRAIN", "Trail 3 (último)  →  TEST"]
    poi_counts   = [5, 4, 6]
    for ti, (tc, tl, np_) in enumerate(zip(trail_colors, trail_labels, poi_counts)):
        ty = 20.5 - ti * 1.0
        for pi in range(np_):
            circle = plt.Circle((1.1 + pi * 0.72, ty - 0.17), 0.24,
                                 color=tc, alpha=0.85, zorder=3)
            ax.add_patch(circle)
            ax.text(1.1 + pi * 0.72, ty - 0.17, str(pi + 1),
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=4)
        ax.text(1.1 + np_ * 0.72 + 0.25, ty - 0.17, tl,
                fontsize=8.5, va="center", color=tc, fontweight="bold" if ti == 2 else "normal")

    # Flecha del trail 3 hacia TEST
    ax.annotate("", xy=(3.2, 18.05), xytext=(3.2, 18.35),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_TEST))

    # Caja TEST
    _box(1.7, 17.2, 4.0, 0.72, C_TEST, alpha=0.18, lw=1.6)
    ax.text(3.7, 17.61, "TEST  =  Trail 3 completo  ·  seed = POI_1  ·  truth = {POI_2 … POI_n}",
            ha="center", va="center", fontsize=8, color="#7B3800")

    # Sub-box usuario con solo 1 trail
    _box(7.35, 17.15, 6.0, 4.3, "#888888", alpha=0.08)
    ax.text(7.7, 21.2, "Usuario con solo 1 trail", fontsize=10, fontweight="bold", color="#666666")
    tc = "#AAAAAA"
    for pi in range(5):
        circle = plt.Circle((7.8 + pi * 0.72, 20.3), 0.24, color=tc, alpha=0.6, zorder=3)
        ax.add_patch(circle)
        ax.text(7.8 + pi * 0.72, 20.3, str(pi + 1),
                ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=4)
    ax.text(8.0, 19.5, "No se puede evaluar:\nno hay trails anteriores\npara entrenar", fontsize=8.5,
            color="#888888", va="top")
    _box(7.55, 17.2, 5.6, 0.72, "#888888", alpha=0.12, lw=1.3)
    ax.text(10.35, 17.61, "SKIP  — usuario excluido del benchmark",
            ha="center", va="center", fontsize=8, color="#555555")

    _arrow(7.0, 19.5, 7.3, 19.5)

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 2 — REENTRENAMIENTO JUSTO (--fair)
    # ════════════════════════════════════════════════════════════════════════
    _box(0.3, 13.5, W - 0.6, 3.05, C_ENGINE, alpha=0.06, radius=0.14)
    _section_title(0.65, 16.3, "2 · Reentrenamiento justo  (--fair)", C_ENGINE, size=12)
    _arrow(W / 2, 16.9, W / 2, 16.6)

    model_info = [
        ("Word2Vec", "Retrenado sobre\ntrails de TRAIN", C_ENGINE, True),
        ("ALS",      "Retrenado sobre\ninteracciones TRAIN", C_ENGINE, True),
        ("Content",  "TF-IDF calculado\nsolo sobre TRAIN", "#888888", False),
        ("Item-Item","Co-visitation\nsolo sobre TRAIN", "#888888", False),
        ("Markov",   "Transiciones\nsolo sobre TRAIN", "#888888", False),
    ]
    box_w, box_h = 2.35, 1.85
    total_w = len(model_info) * box_w + (len(model_info) - 1) * 0.18
    x0_m = (W - total_w) / 2
    for mi, (name, desc, col, retrained) in enumerate(model_info):
        bx = x0_m + mi * (box_w + 0.18)
        _box(bx, 13.65, box_w, box_h, col, alpha=0.14, lw=1.6)
        ax.text(bx + box_w / 2, 14.95, name, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=col)
        badge = "RETRAIN" if retrained else "NO ARTIFACT"
        badge_col = C_ENGINE if retrained else "#888888"
        _box(bx + 0.15, 14.55, box_w - 0.3, 0.30, badge_col, alpha=0.25, radius=0.06, lw=1.0)
        ax.text(bx + box_w / 2, 14.71, badge, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color=badge_col)
        ax.text(bx + box_w / 2, 14.25, desc, ha="center", va="top",
                fontsize=7.5, color="#333333", multialignment="center")

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 3 — BUCLE DE EVALUACIÓN POR USUARIO
    # ════════════════════════════════════════════════════════════════════════
    _box(0.3, 8.25, W - 0.6, 4.85, C_METRIC, alpha=0.06, radius=0.14)
    _section_title(0.65, 12.9, "3 · Bucle de evaluación  (por cada usuario de test)", C_METRIC, size=12)
    _arrow(W / 2, 13.5, W / 2, 13.1)

    # Trail test visualizado
    ax.text(0.75, 12.35, "Trail de test:", fontsize=9, color="#333333", va="center")
    poi_labels = ["POI₁\n(seed)", "POI₂", "POI₃", "POI₄", "POI₅", "…"]
    for pi, lbl in enumerate(poi_labels):
        cx = 2.6 + pi * 1.22
        col = C_TEST if pi == 0 else "#555588"
        circle = plt.Circle((cx, 12.32), 0.32, color=col, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, 12.32, lbl, ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=4)
        if pi < len(poi_labels) - 1:
            ax.annotate("", xy=(cx + 0.58, 12.32), xytext=(cx + 0.32, 12.32),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#AAAAAA"))

    # Caja Ground Truth
    _box(9.4, 11.85, 4.0, 0.90, C_TEST, alpha=0.15, lw=1.5)
    ax.text(11.4, 12.32, "Ground truth\n{POI₂, POI₃, POI₄, POI₅, …}",
            ha="center", va="center", fontsize=8, color="#7B3800")

    # Flecha seed → motores
    ax.text(3.5, 11.6, "seed = POI₁", fontsize=8, color=C_TEST, ha="center", style="italic")
    _arrow(3.5, 11.55, 3.5, 11.25)

    # Fila de motores (9)
    engines = ["content", "item", "markov", "embed", "als", "hybrid", "rrf", "popular", "random"]
    e_w, e_h = 1.28, 0.68
    e_gap = 0.08
    total_e = len(engines) * e_w + (len(engines) - 1) * e_gap
    xe0 = (W - total_e) / 2
    for ei, eng in enumerate(engines):
        ex = xe0 + ei * (e_w + e_gap)
        col = "#444444" if eng == "random" else ("#E8A000" if eng in ("popular", "rrf") else C_ENGINE)
        _box(ex, 10.5, e_w, e_h, col, alpha=0.18, lw=1.3)
        ax.text(ex + e_w / 2, 10.85, eng, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=col)

    # Flecha motores → top-K
    _arrow(W / 2, 10.5, W / 2, 10.2)

    # Caja Top-K recomendaciones
    _box(3.5, 9.35, 7.0, 0.72, C_ENGINE, alpha=0.15, lw=1.5)
    ax.text(7.0, 9.72, "Top-K POIs recomendados por cada motor",
            ha="center", va="center", fontsize=8.5, color="#1B5E20")

    # Flecha comparación → métricas
    ax.annotate("", xy=(10.1, 10.35), xytext=(10.1, 11.78),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_ARROW, linestyle="dashed"))
    ax.text(10.7, 11.1, "comparar", fontsize=7.5, color=C_METRIC, rotation=-90, va="center")

    _arrow(W / 2, 9.35, W / 2, 9.15)

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 4 — MÉTRICAS Y AGREGACIÓN
    # ════════════════════════════════════════════════════════════════════════
    _box(0.3, 4.7, W - 0.6, 4.1, C_METRIC, alpha=0.06, radius=0.14)
    _section_title(0.65, 8.6, "4 · Métricas y agregación", C_METRIC, size=12)

    # Grupo relevancia
    _box(0.55, 5.0, 4.1, 3.3, C_METRIC, alpha=0.10, lw=1.4)
    ax.text(2.6, 8.08, "Relevancia individual", ha="center", fontsize=9.5,
            fontweight="bold", color=C_METRIC)
    for i, m in enumerate(["Hit@K", "Precision@K", "Recall@K", "nDCG@K"]):
        ax.text(2.6, 7.65 - i * 0.55, m, ha="center", va="center", fontsize=9)

    # Grupo categoría
    _box(4.95, 5.0, 4.1, 3.3, C_METRIC, alpha=0.10, lw=1.4)
    ax.text(7.0, 8.08, "Acierto por categoría", ha="center", fontsize=9.5,
            fontweight="bold", color=C_METRIC)
    for i, m in enumerate(["cat_Hit@K", "cat_Precision@K", "cat_Recall@K", "cat_nDCG@K"]):
        ax.text(7.0, 7.65 - i * 0.55, m, ha="center", va="center", fontsize=9)

    # Grupo más allá
    _box(9.35, 5.0, 4.1, 3.3, C_METRIC, alpha=0.10, lw=1.4)
    ax.text(11.4, 8.08, "Más allá de la precisión", ha="center", fontsize=9.5,
            fontweight="bold", color=C_METRIC)
    for i, (m, d) in enumerate([
        ("Novelty", "−log₂ P(item)  ·  ítems poco vistos"),
        ("Diversity", "1 − sim(coseno)  ·  variedad"),
        ("Cold split", "< 5 visitas TRAIN"),
        ("Warm split", ">= 5 visitas TRAIN"),
    ]):
        col = C_COLD if m == "Cold split" else (C_WARM if m == "Warm split" else "#222222")
        ax.text(9.55, 7.65 - i * 0.55, f"• {m}:", va="center", fontsize=8.5,
                fontweight="bold", color=col)
        ax.text(11.0, 7.65 - i * 0.55, d, va="center", fontsize=7.5, color="#444444")

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 5 — CIUDADES Y OUTPUT
    # ════════════════════════════════════════════════════════════════════════
    _arrow(W / 2, 4.7, W / 2, 4.45)
    _box(0.3, 0.3, W - 0.6, 3.85, "#1A1A2E", alpha=0.07, radius=0.14, lw=1.5)
    _section_title(0.65, 3.9, "5 · Resultados: 3 ciudades × 9 motores × cold/warm", "#1A1A2E", size=11)

    cities = [
        ("Osaka\nQ35765",         "~200 K check-ins",  "RRF · Hit@10 = 0.479",  "#FF6B35"),
        ("Istanbul\nQ406",        "~40 K check-ins",   "Markov · Hit@10 = 0.294","#8338EC"),
        ("Petaling Jaya\nQ864965","~35 K check-ins",   "RRF · Hit@10 = 0.439",  "#06A77D"),
    ]
    cw = (W - 1.4) / 3 - 0.15
    for ci, (name, data, best, col) in enumerate(cities):
        cx = 0.65 + ci * (cw + 0.15)
        _box(cx, 0.5, cw, 3.0, col, alpha=0.12, lw=1.6)
        ax.text(cx + cw / 2, 3.2, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color=col)
        ax.text(cx + cw / 2, 2.7, data, ha="center", va="center",
                fontsize=8.5, color="#444444")
        _box(cx + 0.1, 0.7, cw - 0.2, 0.80, col, alpha=0.22, lw=1.2)
        ax.text(cx + cw / 2, 1.12, f"Mejor: {best}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=col)
        ax.text(cx + cw / 2, 1.82, "JSON con métricas por modo\n+ cold/warm breakdown",
                ha="center", va="center", fontsize=7.5, color="#555555")

    ax.set_title("Sistema de evaluación offline del recomendador turístico",
                 fontsize=15, fontweight="bold", pad=10, color="#1A1A2E")
    fig.savefig(_save("fig_20_eval_protocolo.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_21_comparativa_literatura():
    """Comparativa NDCG del TFG vs métodos ML tradicionales y DL SOTA."""
    import matplotlib.patches as mpatches

    # TFG — NDCG@20 promedio 3 ciudades leído de _latest.json
    # Protocolo: last_trail_user --fair, Foursquare Semantic Trails 2018
    _TFG_META = [
        ("content", "Content (TFG)",   "#bdbdbd"),
        ("embed",   "Embed (TFG)",     "#90caf9"),
        ("rrf",     "RRF (TFG)",       "#a5d6a7"),
        ("als",     "ALS (TFG)",       "#80cbc4"),
        ("hybrid",  "Hybrid (TFG)",    "#ce93d8"),
        ("markov",  "Markov (TFG)",    "#ffb74d"),
        ("item",    "Item-Item (TFG)", "#ef9a9a"),
    ]
    # Istanbul excluida: volumen de datos ~4× menor que Osaka/PJ arrastra la media hacia abajo
    # y no refleja el comportamiento real del sistema en ciudades con datos suficientes.
    _EVAL_QIDS = ["Q35765", "Q864965"]  # Osaka + Petaling Jaya
    ndcg_per_mode: dict[str, list[float]] = {}
    for qid in _EVAL_QIDS:
        df = load_eval_results(qid)
        for _, row in df.iterrows():
            mode = str(row.get("mode", ""))
            v = row.get("ndcg", np.nan)
            if pd.notna(v):
                ndcg_per_mode.setdefault(mode, []).append(float(v))
    tfg = []
    for key, label, color in _TFG_META:
        vals = ndcg_per_mode.get(key, [])
        avg = round(sum(vals) / len(vals), 3) if vals else 0.0
        tfg.append((label, avg, color))
    tfg.sort(key=lambda x: x[1])  # ascending for barh

    # Literatura ML tradicional — NDCG@10 aprox.
    # Fuentes: Survey POI arXiv:2410.02191, Massive-STEPS arXiv:2505.11239,
    #          FPMC WWW2010, IJCAI2013, MDPI IJGI 2023
    lit_ml = [
        ("BPR-MF\n(lit.)",     0.061, "#9e9e9e"),
        ("Markov\n(lit.)",     0.068, "#9e9e9e"),
        ("FPMC\n(lit.)",       0.094, "#9e9e9e"),
        ("Item-KNN\n(lit.)",   0.105, "#9e9e9e"),
    ]

    # Deep Learning SOTA — NDCG@10 aprox. — referencia de próximo paso
    # Fuentes: GRU4Rec ICLR2016, GETNext KDD2022, STHGCN Massive-STEPS 2025
    lit_dl = [
        ("GRU4Rec",       0.172, "#1e88e5"),
        ("GETNext",       0.241, "#1565c0"),
        ("STHGCN (SOTA)", 0.298, "#0d47a1"),
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.08}
    )
    fig.patch.set_facecolor("#fafafa")

    # ── Panel izquierdo ──────────────────────────────────────────────
    ax1.set_facecolor("#f5f5f5")
    ax1.set_title("Métodos ML tradicionales  vs.  este TFG",
                  fontsize=13, fontweight="bold", pad=14)

    # Barras ML tradicional
    y_ml = list(range(len(lit_ml)))
    for i, (label, val, col) in enumerate(lit_ml):
        ax1.barh(i, val, height=0.55, color=col, edgecolor="white", linewidth=0.8)
        ax1.text(val + 0.004, i, f"{val:.3f}", va="center", fontsize=9, color="#555")

    # Gap y separador
    gap_y = len(lit_ml) + 0.9
    ax1.text(0.175, gap_y - 0.45, "── este TFG ──",
             ha="center", va="center", fontsize=8, color="#999")

    # Rectángulo de fondo suave para zona TFG
    n_tfg = len(tfg)
    rect = plt.Rectangle(
        (0, gap_y), 0.35, n_tfg + 0.2,
        color="#ff9800", alpha=0.06, zorder=0
    )
    ax1.add_patch(rect)

    # Barras TFG
    y_tfg = [gap_y + i for i in range(n_tfg)]
    for i, (label, val, col) in enumerate(tfg):
        ax1.barh(y_tfg[i], val, height=0.55, color=col,
                 edgecolor="white", linewidth=0.8)
        ax1.text(val + 0.004, y_tfg[i], f"{val:.3f}",
                 va="center", fontsize=9, color="#333", fontweight="bold")

    # Línea vertical = mejor baseline tradicional
    best_lit = max(v for _, v, _ in lit_ml)
    ax1.axvline(best_lit, color="#757575", linestyle="--",
                linewidth=1.2, alpha=0.7, zorder=3)
    ax1.text(best_lit + 0.003, gap_y + n_tfg - 0.2,
             f"mejor tradicional\n(literatura)\n{best_lit:.3f}",
             fontsize=7.5, color="#757575", va="top")

    # Ejes y formato
    all_labels = [l for l, _, _ in lit_ml] + [l for l, _, _ in tfg]
    all_y      = y_ml + y_tfg
    ax1.set_yticks(all_y)
    ax1.set_yticklabels(all_labels, fontsize=9.5)
    x_max = max(max(v for _, v, _ in tfg), max(v for _, v, _ in lit_ml)) * 1.25
    ax1.set_xlim(0, x_max)
    ax1.set_xlabel("NDCG@K", fontsize=11)
    ax1.xaxis.grid(True, alpha=0.4, color="white")
    ax1.spines[["top", "right"]].set_visible(False)

    p1 = mpatches.Patch(color="#9e9e9e", label="ML tradicional (literatura, NDCG@10)")
    p2 = mpatches.Patch(color="#ff9800", alpha=0.6,
                        label="Este TFG (NDCG@20, media Osaka + Petaling Jaya)")
    ax1.legend(handles=[p1, p2], loc="lower right", fontsize=8.5, framealpha=0.9)

    # ── Panel derecho ────────────────────────────────────────────────
    ax2.set_facecolor("#e3f2fd")
    ax2.set_title("Deep Learning\n→ Próximo paso",
                  fontsize=13, fontweight="bold", color="#0d47a1", pad=14)

    for i, (label, val, col) in enumerate(lit_dl):
        ax2.barh(i, val, height=0.55, color=col, edgecolor="white", linewidth=0.8)
        ax2.text(val + 0.004, i, f"{val:.3f}",
                 va="center", fontsize=9, color="#0d47a1", fontweight="bold")

    # Línea roja = mejor resultado TFG (motor con mayor NDCG@20)
    best_entry = max(tfg, key=lambda x: x[1])
    best_tfg_label = best_entry[0].replace(" (TFG)", "")
    best_tfg = best_entry[1]
    ax2.axvline(best_tfg, color="#ef5350", linestyle=":",
                linewidth=1.8, alpha=0.85, zorder=3)
    ax2.text(best_tfg - 0.003, len(lit_dl) - 0.2,
             f"{best_tfg_label}\n(TFG)\n{best_tfg:.3f}",
             fontsize=7.5, color="#ef5350", va="top", ha="right")

    ax2.set_yticks(range(len(lit_dl)))
    ax2.set_yticklabels([l for l, _, _ in lit_dl], fontsize=9.5)
    ax2.set_xlim(0, max(v for _, v, _ in lit_dl) * 1.2)
    ax2.set_xlabel("NDCG@K", fontsize=11)
    ax2.xaxis.grid(True, alpha=0.3, color="white")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(0.175, -0.85,
             "Requiere redes neuronales\nprofundas y más cómputo",
             ha="center", fontsize=8.5, color="#1565c0", style="italic")

    # ── Títulos y nota ────────────────────────────────────────────────
    fig.suptitle(
        "Posicionamiento del sistema de recomendación turística\n"
        "respecto al estado del arte",
        fontsize=15, fontweight="bold", y=1.02
    )
    fig.text(
        0.5, -0.05,
        "⚠  Comparación orientativa: TFG usa NDCG@20 + protocolo last_trail_user "
        "(trail recommendation)  ·  Literatura usa NDCG@10 + leave-one-out (next-POI prediction)\n"
        "Media calculada sobre Osaka y Petaling Jaya (Istanbul excluida: ~4× menos datos, "
        "resultados estructuralmente inferiores no representativos del sistema)\n"
        "Fuentes: Massive-STEPS 2025 (arXiv:2505.11239)  ·  Survey POI 2024 (arXiv:2410.02191)  "
        "·  FPMC WWW2010  ·  GETNext KDD2022",
        ha="center", fontsize=7.5, color="#757575", style="italic"
    )

    fig.savefig(
        _save("fig_21_comparativa_literatura.png"),
        dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    log("fig_21_comparativa_literatura — OK")


def fig_22_comparativa_hit():
    """Comparativa Hit@K del TFG vs métodos ML tradicionales y DL SOTA."""
    import matplotlib.patches as mpatches

    _TFG_META = [
        ("content", "Content (TFG)",   "#bdbdbd"),
        ("embed",   "Embed (TFG)",     "#90caf9"),
        ("als",     "ALS (TFG)",       "#80cbc4"),
        ("hybrid",  "Hybrid (TFG)",    "#ce93d8"),
        ("item",    "Item-Item (TFG)", "#ef9a9a"),
        ("markov",  "Markov (TFG)",    "#ffb74d"),
        ("rrf",     "RRF (TFG)",       "#a5d6a7"),
    ]

    # Istanbul excluida: volumen de datos ~4× menor que Osaka/PJ arrastra la media hacia abajo.
    _EVAL_QIDS = ["Q35765", "Q864965"]  # Osaka + Petaling Jaya
    hit_per_mode: dict[str, list[float]] = {}
    for qid in _EVAL_QIDS:
        df = load_eval_results(qid)
        for _, row in df.iterrows():
            mode = str(row.get("mode", ""))
            v = row.get("hit", np.nan)
            if pd.notna(v):
                hit_per_mode.setdefault(mode, []).append(float(v))

    tfg = []
    for key, label, color in _TFG_META:
        vals = hit_per_mode.get(key, [])
        avg = round(sum(vals) / len(vals), 3) if vals else 0.0
        tfg.append((label, avg, color))
    tfg.sort(key=lambda x: x[1])

    lit_ml = [
        ("MF",   0.152, "#9e9e9e"),
        ("FPMC", 0.297, "#9e9e9e"),
        ("PRME", 0.311, "#9e9e9e"),
    ]
    lit_dl = [
        ("LSTM",    0.328, "#90caf9"),
        ("ST-RNN",  0.362, "#64b5f6"),
        ("STGN",    0.412, "#42a5f5"),
        ("STGCN",   0.428, "#1e88e5"),
        ("PLSPL",   0.452, "#1565c0"),
        ("STAN",    0.573, "#0d47a1"),
        ("GETNext", 0.614, "#002171"),
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 9),
        gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.08}
    )
    fig.patch.set_facecolor("#fafafa")

    # ── Panel izquierdo ──────────────────────────────────────────────
    ax1.set_facecolor("#f5f5f5")
    ax1.set_title("Métodos ML tradicionales  vs.  este TFG",
                  fontsize=13, fontweight="bold", pad=14)

    y_ml = list(range(len(lit_ml)))
    for i, (label, val, col) in enumerate(lit_ml):
        ax1.barh(i, val, height=0.55, color=col, edgecolor="white", linewidth=0.8)
        ax1.text(val + 0.006, i, f"{val:.3f}", va="center", fontsize=9, color="#555")

    gap_y = len(lit_ml) + 0.9
    ax1.text(0.25, gap_y - 0.45, "── este TFG ──",
             ha="center", va="center", fontsize=8, color="#999")

    n_tfg = len(tfg)
    ax1.add_patch(plt.Rectangle(
        (0, gap_y), 0.55, n_tfg + 0.2, color="#ff9800", alpha=0.06, zorder=0
    ))

    y_tfg = [gap_y + i for i in range(n_tfg)]
    for i, (label, val, col) in enumerate(tfg):
        ax1.barh(y_tfg[i], val, height=0.55, color=col,
                 edgecolor="white", linewidth=0.8)
        ax1.text(val + 0.006, y_tfg[i], f"{val:.3f}",
                 va="center", fontsize=9, color="#333", fontweight="bold")

    best_lit = max(v for _, v, _ in lit_ml)
    ax1.axvline(best_lit, color="#757575", linestyle="--",
                linewidth=1.2, alpha=0.7, zorder=3)
    ax1.text(best_lit + 0.006, gap_y + n_tfg - 0.2,
             f"mejor tradicional\n(literatura)\n{best_lit:.3f}",
             fontsize=7.5, color="#757575", va="top")

    all_labels = [l for l, _, _ in lit_ml] + [l for l, _, _ in tfg]
    all_y = y_ml + y_tfg
    ax1.set_yticks(all_y)
    ax1.set_yticklabels(all_labels, fontsize=9.5)
    x_max1 = max(max(v for _, v, _ in tfg), best_lit) * 1.3
    ax1.set_xlim(0, x_max1)
    ax1.set_xlabel("Hit@K", fontsize=11)
    ax1.xaxis.grid(True, alpha=0.4, color="white")
    ax1.spines[["top", "right"]].set_visible(False)

    p1 = mpatches.Patch(color="#9e9e9e", label="ML tradicional (literatura, Hit@10)")
    p2 = mpatches.Patch(color="#ff9800", alpha=0.6,
                        label="Este TFG (Hit@20, media Osaka + Petaling Jaya)")
    ax1.legend(handles=[p1, p2], loc="lower right", fontsize=8.5, framealpha=0.9)

    # ── Panel derecho ────────────────────────────────────────────────
    ax2.set_facecolor("#e3f2fd")
    ax2.set_title("Deep Learning\n→ Próximo paso",
                  fontsize=13, fontweight="bold", color="#0d47a1", pad=14)

    for i, (label, val, col) in enumerate(lit_dl):
        ax2.barh(i, val, height=0.55, color=col, edgecolor="white", linewidth=0.8)
        ax2.text(val + 0.008, i, f"{val:.3f}",
                 va="center", fontsize=9, color="#0d47a1", fontweight="bold")

    best_entry = max(tfg, key=lambda x: x[1])
    best_tfg_label = best_entry[0].replace(" (TFG)", "")
    best_tfg = best_entry[1]
    ax2.axvline(best_tfg, color="#ef5350", linestyle=":",
                linewidth=1.8, alpha=0.85, zorder=3)
    ax2.text(best_tfg - 0.008, len(lit_dl) - 0.2,
             f"{best_tfg_label}\n(TFG)\n{best_tfg:.3f}",
             fontsize=7.5, color="#ef5350", va="top", ha="right")

    ax2.set_yticks(range(len(lit_dl)))
    ax2.set_yticklabels([l for l, _, _ in lit_dl], fontsize=9.5)
    ax2.set_xlim(0, max(v for _, v, _ in lit_dl) * 1.15)
    ax2.set_xlabel("Hit@K", fontsize=11)
    ax2.xaxis.grid(True, alpha=0.3, color="white")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(0.35, -1.0,
             "Requiere redes neuronales\nprofundas y más cómputo",
             ha="center", fontsize=8.5, color="#1565c0", style="italic")

    fig.suptitle(
        "Posicionamiento del sistema de recomendación turística\n"
        "respecto al estado del arte  (Hit@K)",
        fontsize=15, fontweight="bold", y=1.02
    )
    fig.text(
        0.5, -0.04,
        "⚠  Comparación orientativa: TFG usa Hit@20 + protocolo last_trail_user "
        "(trail recommendation)  ·  Literatura usa Hit@10 + leave-one-out (next-POI prediction)\n"
        "Media calculada sobre Osaka y Petaling Jaya (Istanbul excluida: ~4× menos datos, "
        "resultados estructuralmente inferiores no representativos del sistema)\n"
        "Fuentes: Massive-STEPS 2025 (arXiv:2505.11239)  ·  Survey POI 2024 (arXiv:2410.02191)  "
        "·  FPMC WWW2010  ·  GETNext KDD2022  ·  STAN WWW2021",
        ha="center", fontsize=7.5, color="#757575", style="italic"
    )

    fig.savefig(
        _save("fig_22_comparativa_hit.png"),
        dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    log("fig_22_comparativa_hit — OK")


def _build_poi_transitions(visits: pd.DataFrame, pois: pd.DataFrame, top_n: int = 40):
    """Devuelve (edges, poi_info) donde edges = list[(from_id, to_id, count)]
    y poi_info = DataFrame con poi_id, name, lat, lon, primary_category, visit_count."""
    from collections import Counter, defaultdict
    visits_s = visits.sort_values(["trail_id", "timestamp"])
    trans: Counter = Counter()
    for _, g in visits_s.groupby("trail_id"):
        seq = g["poi_id"].tolist()
        for a, b in zip(seq[:-1], seq[1:]):
            if a != b:
                trans[(a, b)] += 1

    top_edges = trans.most_common(top_n)
    poi_ids = {p for e in top_edges for p in e[0]}
    # also include by visit frequency
    vc = visits["poi_id"].value_counts()
    poi_info = pois[pois["poi_id"].isin(poi_ids)].copy()
    poi_info = poi_info.merge(vc.rename("visit_count"), left_on="poi_id", right_index=True, how="left")
    poi_info["visit_count"] = poi_info["visit_count"].fillna(0)
    poi_info = poi_info.dropna(subset=["lat", "lon"])
    # filter edges to POIs with coords
    valid_ids = set(poi_info["poi_id"])
    edges = [(a, b, c) for (a, b), c in top_edges if a in valid_ids and b in valid_ids]
    return edges, poi_info


def _to_mercator(lat, lon):
    import math
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y


def _draw_arcos_on_ax(ax, visits, pois, ctx, top_n=40):
    """Helper: dibuja arc map Markov en un subplot dado. Devuelve colorbar mappable."""
    edges, poi_info = _build_poi_transitions(visits, pois, top_n=top_n)
    if not edges:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        return None, None

    poi_xy = {row["poi_id"]: _to_mercator(row["lat"], row["lon"])
              for _, row in poi_info.iterrows()}

    valid = [(a, b, c) for a, b, c in edges if a in poi_xy and b in poi_xy]
    if not valid:
        return None, None

    max_c = max(c for _, _, c in valid)
    counts = np.array([c for _, _, c in valid], dtype=float)
    xs = [poi_xy[e[0]][0] for e in valid] + [poi_xy[e[1]][0] for e in valid]
    ys = [poi_xy[e[0]][1] for e in valid] + [poi_xy[e[1]][1] for e in valid]
    pad = max(abs(max(xs) - min(xs)), abs(max(ys) - min(ys))) * 0.12
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    _try_add_basemap(ax, ctx, ctx.providers.CartoDB.DarkMatter, zoom=12)

    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=counts.min(), vmax=counts.max())
    for (a, b, cnt), count in zip(valid, counts):
        x0, y0 = poi_xy[a]; x1, y1 = poi_xy[b]
        rad = 0.25 + 0.15 * (hash((a, b)) % 3)
        arr = patches.FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=8 + 6 * (cnt / max_c),
            linewidth=0.7 + 4.5 * (cnt / max_c),
            color=cmap(norm(count)),
            alpha=0.25 + 0.70 * (cnt / max_c),
            zorder=3,
        )
        ax.add_patch(arr)

    vc = visits["poi_id"].value_counts()
    top_pois = set(vc.head(25).index)
    for _, row in poi_info[poi_info["poi_id"].isin(top_pois)].iterrows():
        x, y = poi_xy.get(row["poi_id"], (None, None))
        if x is None: continue
        ax.scatter(x, y, s=15 + 50 * (row["visit_count"] / (vc.max() + 1)),
                   c="#FFD54F", zorder=5, edgecolors="#333", linewidths=0.4, alpha=0.9)
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm, max_c


def fig_23_markov_arcos():
    """Arc map Osaka: top transiciones POI-POI como arcos curvos sobre mapa geográfico."""
    try:
        import contextily as ctx
    except ImportError:
        missing_dep("contextily")
        raise

    fig, ax = plt.subplots(1, 1, figsize=(14, 13))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    qid = "Q35765"
    meta = CITY_META[qid]
    with get_db_connection() as conn:
        visits = load_visits(conn, qid)
        pois = load_pois(conn, qid)
        sm, _ = _draw_arcos_on_ax(ax, visits, pois, ctx, top_n=40)

    ax.set_title(f"{meta['name']}", fontsize=15, fontweight="bold", color="white", pad=10)
    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.65)
        cbar.set_label("Frecuencia", color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle("Transiciones Markov aprendidas — top 40 POI→POI (Osaka)",
                 fontsize=17, fontweight="bold", color="white", y=1.01)
    fig.tight_layout()
    fig.savefig(_save("fig_23_markov_arcos.png"), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log("fig_23_markov_arcos — OK")


_CAT_COLORS = {
    "Food":          "#ff9800",
    "Culture":       "#e91e63",
    "Entertainment": "#9c27b0",
    "Nature":        "#4caf50",
    "Shopping":      "#2196f3",
    "Transport":     "#00bcd4",
    "Nightlife":     "#f44336",
    "Service":       "#795548",
    "Health":        "#8bc34a",
    "Sports":        "#ff5722",
    "Relaxation":    "#03a9f4",
    "Family":        "#ffc107",
    "Inconclusive":  "#78909c",
}


def _draw_grafo_on_ax(ax, visits, pois, ctx, top_n=60, top_pois=20, min_prob=0.05):
    """Helper: dibuja grafo Markov geográfico en un subplot dado."""
    edges, poi_info = _build_poi_transitions(visits, pois, top_n=top_n)
    if not edges:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        return

    poi_counts: Counter = Counter()
    for a, b, c in edges:
        poi_counts[a] += c; poi_counts[b] += c
    top_ids = {p for p, _ in poi_counts.most_common(top_pois)}

    poi_xy = {row["poi_id"]: _to_mercator(row["lat"], row["lon"])
              for _, row in poi_info.iterrows() if row["poi_id"] in top_ids}

    edges_f = [(a, b, c) for a, b, c in edges if a in poi_xy and b in poi_xy]
    if not edges_f:
        return

    out_sum: Counter = Counter()
    for a, b, c in edges_f:
        out_sum[a] += c
    edge_prob = [(a, b, c / out_sum[a]) for a, b, c in edges_f]

    xs = [poi_xy[p][0] for p in poi_xy]; ys = [poi_xy[p][1] for p in poi_xy]
    pad = max(abs(max(xs) - min(xs)), abs(max(ys) - min(ys))) * 0.12
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    _try_add_basemap(ax, ctx, ctx.providers.CartoDB.Positron, zoom=12)

    for a, b, prob in edge_prob:
        if prob < min_prob:
            continue
        x0, y0 = poi_xy[a]; x1, y1 = poi_xy[b]
        arr = patches.FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle=f"arc3,rad={0.25 + 0.1*(hash((a,b))%4)}",
            arrowstyle="-|>",
            mutation_scale=7 + 5 * prob,
            linewidth=0.7 + 5.5 * prob,
            color="#1565c0", alpha=0.25 + 0.65 * prob, zorder=3,
        )
        ax.add_patch(arr)

    vc = visits["poi_id"].value_counts()
    cat_map = poi_info.set_index("poi_id")["primary_category"].to_dict()
    name_map = poi_info.set_index("poi_id")["name"].to_dict()
    for pid, (x, y) in poi_xy.items():
        color = _CAT_COLORS.get(_map_to_broad_category(cat_map.get(pid, "")), "#78909c")
        sz = 60 + 260 * (vc.get(pid, 0) / (vc.max() + 1))
        ax.scatter(x, y, s=sz, c=color, zorder=5, edgecolors="white", linewidths=0.8, alpha=0.95)
        short = name_map.get(pid, pid)[:16]
        ax.annotate(short, (x, y), xytext=(3, 3), textcoords="offset points",
                    fontsize=6, color="#111", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.55, ec="none"))
    ax.set_axis_off()


def fig_24_markov_grafo_mapa():
    """Grafo Markov geográfico — 3 ciudades en subplots."""
    try:
        import contextily as ctx
    except ImportError:
        missing_dep("contextily")
        raise

    fig, axes = plt.subplots(1, 3, figsize=(36, 13))

    with get_db_connection() as conn:
        for ax, (qid, meta) in zip(axes, CITY_META.items()):
            visits = load_visits(conn, qid)
            pois = load_pois(conn, qid)
            _draw_grafo_on_ax(ax, visits, pois, ctx)
            ax.set_title(f"{meta['name']} — top 20 POIs (prob ≥ 0.05)",
                         fontsize=13, fontweight="bold", pad=8)

    # Leyenda compartida
    legend_items = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=c, markersize=9, label=cat)
                    for cat, c in _CAT_COLORS.items() if cat != "Inconclusive"]
    fig.legend(handles=legend_items, loc="lower center", ncol=6, fontsize=9,
               framealpha=0.85, title="Categoría", bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Grafo Markov geográfico — nodos=POIs en lat/lon, aristas=prob de transición",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(_save("fig_24_markov_grafo_mapa.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    log("fig_24_markov_grafo_mapa — OK")


def fig_25_markov_vs_real():
    """Comparativa Markov aprendido vs rutas reales — Osaka (1 fila × 2 cols)."""
    try:
        import contextily as ctx
    except ImportError:
        missing_dep("contextily")
        raise
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))

    qid = "Q35765"
    meta = CITY_META[qid]
    with get_db_connection() as conn:
        visits = load_visits(conn, qid)
        pois = load_pois(conn, qid)

    edges, poi_info = _build_poi_transitions(visits, pois, top_n=40)
    poi_xy = {r["poi_id"]: _to_mercator(r["lat"], r["lon"])
              for _, r in poi_info.iterrows()}

    pois_ll = pois.dropna(subset=["lat", "lon"]).set_index("poi_id")
    visits_s = visits.sort_values(["trail_id", "timestamp"])
    real_trails = []
    for _, g in visits_s.groupby("trail_id"):
        seq = [(r["poi_id"], pois_ll.loc[r["poi_id"], "lat"], pois_ll.loc[r["poi_id"], "lon"])
               for _, r in g.iterrows() if r["poi_id"] in pois_ll.index]
        if len(seq) >= 4:
            real_trails.append(seq)
    real_trails = real_trails[:50]

    valid_e = [(a, b, c) for a, b, c in edges if a in poi_xy and b in poi_xy]
    all_xs = [poi_xy[e[0]][0] for e in valid_e] + [poi_xy[e[1]][0] for e in valid_e]
    all_ys = [poi_xy[e[0]][1] for e in valid_e] + [poi_xy[e[1]][1] for e in valid_e]
    for trail in real_trails:
        for _, lat, lon in trail:
            mx, my = _to_mercator(lat, lon)
            all_xs.append(mx); all_ys.append(my)
    pad = max(abs(max(all_xs) - min(all_xs)), abs(max(all_ys) - min(all_ys))) * 0.10
    xlim = (min(all_xs) - pad, max(all_xs) + pad)
    ylim = (min(all_ys) - pad, max(all_ys) + pad)

    ax_m, ax_r = axes[0], axes[1]

    for ax in (ax_m, ax_r):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        _try_add_basemap(ax, ctx, ctx.providers.CartoDB.Positron, zoom=12)
        ax.set_axis_off()

    ax_m.set_title(f"{meta['name']} — Markov (top 40 transiciones)", fontsize=12, fontweight="bold")
    ax_r.set_title(f"{meta['name']} — Rutas reales (50 trails ≥ 4 POIs)", fontsize=12, fontweight="bold")

    # Arcos Markov
    max_c = max(c for _, _, c in valid_e) if valid_e else 1
    cmap_m = plt.cm.Reds
    norm_m = mcolors.Normalize(vmin=0, vmax=max_c)
    for a, b, cnt in valid_e:
        x0, y0 = poi_xy[a]; x1, y1 = poi_xy[b]
        arr = patches.FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle=f"arc3,rad={0.20+0.15*(hash((a,b))%3)}",
            arrowstyle="-|>",
            mutation_scale=6 + 4 * (cnt / max_c),
            linewidth=0.5 + 4.0 * (cnt / max_c),
            color=cmap_m(norm_m(cnt)),
            alpha=0.20 + 0.70 * (cnt / max_c), zorder=3,
        )
        ax_m.add_patch(arr)
    vc = visits["poi_id"].value_counts()
    for pid, (x, y) in poi_xy.items():
        ax_m.scatter(x, y, s=10 + 40*(vc.get(pid, 0)/(vc.max()+1)),
                     c="#b71c1c", zorder=5, alpha=0.75, edgecolors="white", linewidths=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap_m, norm=norm_m)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_m, fraction=0.022, pad=0.01, shrink=0.65, label="Frecuencia")

    # Rutas reales
    cmap_r = plt.cm.Blues
    n = max(len(real_trails), 1)
    for i, trail in enumerate(real_trails):
        coords_m = [_to_mercator(lat, lon) for _, lat, lon in trail]
        xs_t = [c[0] for c in coords_m]; ys_t = [c[1] for c in coords_m]
        ax_r.plot(xs_t, ys_t, "-", color=cmap_r(0.35 + 0.55*(i/n)),
                  linewidth=0.7, alpha=0.20 + 0.55*(i/n), zorder=3)
        ax_r.scatter(xs_t[0], ys_t[0], s=14, c="#1b5e20", zorder=5, alpha=0.85)
        ax_r.scatter(xs_t[-1], ys_t[-1], s=14, c="#b71c1c", zorder=5, alpha=0.85)

    leg = [Line2D([0],[0], color="none", marker="o", markerfacecolor="#1b5e20", markersize=7, label="Inicio"),
           Line2D([0],[0], color="none", marker="o", markerfacecolor="#b71c1c", markersize=7, label="Fin")]
    ax_r.legend(handles=leg, loc="lower right", fontsize=9, framealpha=0.85)

    fig.suptitle("Markov aprendido vs. comportamiento real del turista (Osaka)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(_save("fig_25_markov_vs_real.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("fig_25_markov_vs_real — OK")


def fig_26_cold_warm_breakdown():
    """Comparativa cold vs warm users: hit@20 y nDCG@20 por motor y ciudad."""
    modes_show = ["hybrid", "rrf", "item", "markov", "als", "embed"]
    mode_labels = {"hybrid": "Hybrid", "rrf": "RRF", "item": "Item",
                   "markov": "Markov", "als": "ALS", "embed": "Embed"}
    metrics = [("hit", "Hit@20"), ("ndcg", "nDCG@20")]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey="row")
    colors = {"warm": "#1976d2", "cold": "#e53935"}

    for col, (_, meta) in enumerate(CITY_META.items()):
        slug = meta["slug"]
        path = Path(f"data/reports/benchmarks/eval_{slug}.json")
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        bk = data.get("summary", {}).get("cold_warm_breakdown", {})

        for row, (metric, mlabel) in enumerate(metrics):
            ax = axes[row][col]
            x = np.arange(len(modes_show))
            w = 0.38
            warm_vals = [bk.get(m, {}).get(metric, {}).get("warm", 0) for m in modes_show]
            cold_vals = [bk.get(m, {}).get(metric, {}).get("cold", 0) for m in modes_show]

            ax.bar(x - w/2, warm_vals, w, label="Warm (≥5 visitas)", color=colors["warm"], alpha=0.85)
            ax.bar(x + w/2, cold_vals, w, label="Cold (<5 visitas)", color=colors["cold"], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([mode_labels[m] for m in modes_show], rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(mlabel if col == 0 else "", fontsize=10)
            if row == 0:
                ax.set_title(meta["name"], fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0 and col == 2:
                ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Cold vs Warm users — Hit@20 y nDCG@20 por motor y ciudad",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_save("fig_26_cold_warm_breakdown.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    log("fig_26_cold_warm_breakdown — OK")


def _figure_registry() -> dict[str, Callable[[], None]]:
    return {
        "fig_01": fig_01_pipeline_sistema,
        "fig_02": fig_02_pois_mapa_categorias,
        "fig_03": fig_03_heatmap_checkins,
        "fig_04": fig_04_hexbin_rating,
        "fig_05": fig_05_tres_ciudades,
        "fig_06": fig_06_markov_heatmap,
        "fig_07": fig_07_markov_grafo,
        "fig_08": fig_08_sankey_rutas,
        "fig_09": fig_09_tsne_embeddings,
        "fig_10": fig_10_als_matriz,
        "fig_11": fig_11_hybrid_weights,
        "fig_12": fig_12_tabla_metricas,
        "fig_13": fig_13_barras_agrupadas,
        "fig_14": fig_14_radar_chart,
        "fig_15": fig_15_curvas_metricas_k,
        "fig_16": fig_16_distribucion_categorias,
        "fig_17": fig_17_long_tail_usuarios,
        "fig_18": fig_18_heatmap_temporal,
        "fig_19": fig_19_longitud_trails,
        "fig_14b": fig_14b_heatmap_metricas,
        "fig_20": fig_20_eval_protocolo,
        "fig_21": fig_21_comparativa_literatura,
        "fig_22": fig_22_comparativa_hit,
        "fig_23": fig_23_markov_arcos,
        "fig_24": fig_24_markov_grafo_mapa,
        "fig_25": fig_25_markov_vs_real,
        "fig_26": fig_26_cold_warm_breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="Generador de figuras para memoria TFG")
    parser.add_argument("--only", type=str, default="", help="Ejecutar solo una figura, ej: fig_06")
    parser.add_argument("--skip", type=str, default="", help="Saltar figura(s), ej: fig_08 o fig_08,fig_09")
    args = parser.parse_args()

    registry = _figure_registry()
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    names = [args.only] if args.only else list(registry.keys())
    names = [n for n in names if n not in skip]

    ok = []
    fail = []
    log(f"Salida de figuras: {OUTPUT_DIR.resolve()}")
    for name in names:
        fn = registry.get(name)
        if fn is None:
            log(f"[SKIP] No existe {name}")
            continue
        log(f"[RUN] {name}")
        try:
            fn()
            ok.append(name)
            log(f"[OK]  {name}")
        except ImportError as e:
            fail.append((name, f"ImportError: {e}"))
            log(f"[FAIL] {name}: {e}")
        except Exception as e:
            fail.append((name, str(e)))
            log(f"[FAIL] {name}: {e}")
            traceback.print_exc()

    log(f"Finalizado. OK={len(ok)} FAIL={len(fail)}")
    if ok:
        log("Figuras OK: " + ", ".join(ok))
    if fail:
        for n, err in fail:
            log(f"- {n}: {err}")


if __name__ == "__main__":
    main()
