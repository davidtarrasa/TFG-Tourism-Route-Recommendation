"""
Generador de figuras de dataset/esquema para la memoria TFG.

Figuras producidas en data/reports/figures/dataset/:
  fig_er_diagram.png       — diagrama entidad-relación PostgreSQL
  fig_etl_flow.png         — pipeline ETL del CSV al DB
  fig_bubble_dataset.png   — bubble chart: volumen + categorías por ciudad
  fig_heatmap_coverage.png — heatmap ciudad × estadísticas del dataset
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import psycopg

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from recommender.category_intents import classify_category_intent
    _HAS_INTENTS = True
except Exception:
    _HAS_INTENTS = False
    INCONCLUSIVE = "inconclusive"

OUTPUT_DIR = Path("data/reports/figures/dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITY_META = {
    "Q35765":  {"name": "Osaka",        "color": "#e91e63"},
    "Q406":    {"name": "Istanbul",      "color": "#1565c0"},
    "Q864965": {"name": "Petaling Jaya", "color": "#2e7d32"},
}

CAT_COLORS = {
    "Food": "#ff9800", "Culture": "#e91e63", "Entertainment": "#9c27b0",
    "Nature": "#4caf50", "Shopping": "#2196f3", "Transport": "#00bcd4",
    "Nightlife": "#f44336", "Service": "#795548", "Health": "#8bc34a",
    "Sports": "#ff5722", "Relaxation": "#03a9f4", "Family": "#ffc107",
    "Inconclusive": "#78909c",
}


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}")


def get_db_connection():
    dsn = os.getenv("POSTGRES_DSN", "postgresql://tfg:tfgpass@localhost:55432/tfg_routes")
    return psycopg.connect(dsn)


def _save(name: str) -> Path:
    return OUTPUT_DIR / name


_INTENT_DISPLAY = {
    "food": "Food", "culture": "Culture", "nature": "Nature",
    "nightlife": "Nightlife", "shopping": "Shopping", "service": "Service",
    "health": "Health", "entertainment": "Entertainment",
    "transport": "Transport", "relaxation": "Relaxation",
    "family": "Family", "sports": "Sports",
}


def _map_broad(cat: str) -> str:
    if not _HAS_INTENTS:
        return "Inconclusive"
    raw = str(cat or "").strip()
    if not raw:
        return "Inconclusive"
    try:
        intent, _, _ = classify_category_intent(raw, use_semantic=False)
        return _INTENT_DISPLAY.get(intent, "Inconclusive")
    except Exception:
        return "Inconclusive"


def _visit_col(conn) -> tuple[str, str]:
    """Return (city_col, poi_col) names for visits table."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='visits'"
        )
        cols = {r[0] for r in cur.fetchall()}
    city = "venue_city" if "venue_city" in cols else "city_qid"
    poi = "venue_id" if "venue_id" in cols else "poi_id"
    return city, poi


def _query_city_stats(conn) -> dict:
    city_col, _ = _visit_col(conn)
    stats: dict = {}
    for qid in CITY_META:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT trail_id) "
                f"FROM visits WHERE {city_col} = %s",
                (qid,),
            )
            n_visits, n_users, n_trails = cur.fetchone()

            cur.execute(
                f"SELECT COALESCE(AVG(cnt),0) FROM "
                f"(SELECT trail_id, COUNT(*) AS cnt FROM visits "
                f"WHERE {city_col} = %s GROUP BY trail_id) sub",
                (qid,),
            )
            avg_trail = float(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM pois WHERE city_qid = %s", (qid,))
            n_pois = int(cur.fetchone()[0])

        sparsity = 1.0 - n_visits / max(int(n_users) * n_pois, 1)
        stats[qid] = {
            "n_visits":  int(n_visits),
            "n_users":   int(n_users),
            "n_pois":    n_pois,
            "n_trails":  int(n_trails),
            "avg_trail": avg_trail,
            "sparsity":  float(sparsity),
        }
    return stats


def _query_cat_dist(conn, qid: str) -> dict[str, int]:
    city_col, poi_col = _visit_col(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT p.primary_category, COUNT(*) "
            f"FROM visits v JOIN pois p ON v.{poi_col}::text = p.fsq_id::text "
            f"WHERE v.{city_col} = %s GROUP BY p.primary_category",
            (qid,),
        )
        rows = cur.fetchall()
    dist: dict[str, int] = {}
    for cat, cnt in rows:
        broad = _map_broad(cat or "")
        dist[broad] = dist.get(broad, 0) + cnt
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: ER Diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig_er_diagram():
    """Diagrama Entidad-Relación del esquema PostgreSQL (4 tablas)."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#f5f7fa")
    ax.set_facecolor("#f5f7fa")

    TW = 3.9      # table width
    RH = 0.40     # row height
    HH = 0.58     # header height

    TABLES = {
        "visits": {
            "pos": (0.4, 9.5),
            "color": "#1565c0",
            "columns": [
                ("trail_id",      "INTEGER"),
                ("user_id",       "INTEGER"),
                ("venue_id",      "TEXT  →FK→ pois"),
                ("timestamp",     "TIMESTAMPTZ"),
                ("city_qid",      "TEXT"),
                ("trail_id_orig", "TEXT"),
                ("user_id_orig",  "TEXT"),
            ],
        },
        "pois": {
            "pos": (6.05, 9.5),
            "color": "#6a1b9a",
            "columns": [
                ("fsq_id",           "TEXT  (PK)"),
                ("name",             "TEXT"),
                ("lat / lon",        "DOUBLE PRECISION"),
                ("city_qid",         "TEXT"),
                ("rating",           "DOUBLE PRECISION"),
                ("price_tier",       "INTEGER"),
                ("primary_category", "TEXT"),
                ("total_ratings",    "INTEGER"),
                ("is_free",          "BOOLEAN"),
            ],
        },
        "poi_categories": {
            "pos": (11.7, 9.5),
            "color": "#00695c",
            "columns": [
                ("fsq_id",        "TEXT  (FK → pois)"),
                ("category_id",   "TEXT"),
                ("category_name", "TEXT"),
                ("PK",            "(fsq_id, category_id)"),
            ],
        },
        "saved_routes": {
            "pos": (6.05, 3.8),
            "color": "#b71c1c",
            "columns": [
                ("id",         "BIGSERIAL (PK)"),
                ("user_id",    "BIGINT"),
                ("city_qid",   "TEXT"),
                ("route_type", "TEXT"),
                ("source",     "TEXT"),
                ("payload",    "JSONB"),
                ("created_at", "TIMESTAMPTZ"),
            ],
        },
    }

    bounds: dict[str, tuple] = {}

    def draw_table(name: str, spec: dict):
        x0, y_top = spec["pos"]
        cols = spec["columns"]
        color = spec["color"]
        total_h = HH + len(cols) * RH + 0.15

        # drop shadow
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0 + 0.07, y_top - total_h - 0.07), TW, total_h,
            boxstyle="round,pad=0.05", linewidth=0, facecolor="#c0c0c0", zorder=1))
        # header
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y_top - HH), TW, HH,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor=color, facecolor=color, zorder=2))
        ax.text(x0 + TW / 2, y_top - HH / 2, name,
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color="white", family="monospace", zorder=3)
        # body
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y_top - total_h), TW, total_h - HH,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor=color, facecolor="white", zorder=2))
        for i, (col_name, col_type) in enumerate(cols):
            y = y_top - HH - 0.12 - (i + 0.5) * RH
            if i % 2 == 0:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0 + 0.04, y - RH / 2), TW - 0.08, RH,
                    boxstyle="square,pad=0", linewidth=0,
                    facecolor="#f0f0f0", zorder=2.5))
            is_pk = "(PK)" in col_type
            is_fk = "FK" in col_type or "→FK→" in col_type
            nc = "#1b5e20" if is_pk else ("#4a148c" if is_fk else "#212121")
            fw = "bold" if (is_pk or is_fk) else "normal"
            ax.text(x0 + 0.18, y, col_name,
                    ha="left", va="center", fontsize=8.0, color=nc,
                    fontweight=fw, family="monospace", zorder=3)
            ax.text(x0 + TW - 0.1, y, col_type,
                    ha="right", va="center", fontsize=7.5, color="#555555",
                    family="monospace", zorder=3)

        bounds[name] = (x0, y_top, x0 + TW, y_top - total_h)

    for name, spec in TABLES.items():
        draw_table(name, spec)

    def rel(x0, y0, x1, y1, cs="", ce="", dashed=False):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>", color="#444444", lw=1.8,
                        linestyle="dashed" if dashed else "solid"), zorder=4)
        horiz = abs(x1 - x0) > abs(y1 - y0)
        if cs:
            dx, dy = (-0.22, 0.2) if horiz else (0.18, -0.1)
            ax.text(x0 + dx, y0 + dy, cs, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#c62828", zorder=5)
        if ce:
            dx, dy = (0.22, 0.2) if horiz else (-0.18, 0.1)
            ax.text(x1 + dx, y1 + dy, ce, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#1565c0", zorder=5)

    # visits → pois (N:1)
    rel(bounds["visits"][2], 7.5, bounds["pois"][0], 7.5, cs="N", ce="1")
    # pois → poi_categories (1:N)
    rel(bounds["pois"][2], 8.0, bounds["poi_categories"][0], 8.0, cs="1", ce="N")
    # pois → saved_routes (dashed, conceptual via city_qid)
    cx = bounds["pois"][0] + TW / 2
    rel(cx, bounds["pois"][3], cx, bounds["saved_routes"][1] + 0.02, dashed=True)
    mid_y = (bounds["pois"][3] + bounds["saved_routes"][1]) / 2
    ax.text(cx + 0.2, mid_y, "city_qid", fontsize=8, color="#888888",
            style="italic", va="center")

    legend_items = [
        mpatches.Patch(facecolor="#1b5e20", label="PK — Clave primaria"),
        mpatches.Patch(facecolor="#4a148c", label="FK — Clave foránea"),
        mpatches.Patch(facecolor="none", edgecolor="#888888",
                       linewidth=1, label="Relación conceptual (dashed)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("Diagrama Entidad-Relación — Esquema PostgreSQL",
                 fontsize=14, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(_save("fig_er_diagram.png"), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log("fig_er_diagram — OK")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: ETL Flow
# ─────────────────────────────────────────────────────────────────────────────

def fig_etl_flow():
    """Diagrama de flujo del pipeline ETL (CSV → PostgreSQL)."""
    fig, ax = plt.subplots(figsize=(13, 16))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 16)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    BW = 5.2   # box width
    BH = 1.05  # box height
    CX = 4.5   # center x of main column
    FX = 10.2  # center x of output file labels

    # Steps: (y_center, title, subtitle, color, output_label)
    STEPS = [
        (15.0, "Foursquare Dataset", "raw CSV — ~290 K check-ins · 3 ciudades",
         "#546e7a", ""),
        (13.3, "01_clean_std.py", "Limpieza y normalización de check-ins",
         "#1565c0", "→ std_clean.csv"),
        (11.6, "02_extract_ids.py", "Extracción de IDs únicos de venues",
         "#1565c0", "→ venue_ids.txt"),
        (9.9,  "03_fetch_pois.py", "Consulta API Foursquare (metadata POIs)",
         "#6a1b9a", "→ pois_raw.json"),
        (8.2,  "04_normalize_pois.py", "Normalización y alineación de campos",
         "#1565c0", "→ pois_normalized.json"),
        (6.5,  "05_label_categories.py", "Etiquetado de categorías y precios",
         "#1565c0", "→ category_price_labels.json"),
        (4.8,  "06_impute_pois.py", "Imputación de datos faltantes (×3 ciudades)",
         "#1565c0", "→ pois_enriched_{QID}.json"),
        (3.1,  "07_diagnostics.py", "Diagnóstico y validación de calidad",
         "#37474f", "→ informe diagnóstico"),
        (1.4,  "08_load_postgres.py", "Carga a PostgreSQL (visits · pois · poi_categories)",
         "#1b5e20", ""),
    ]



    def box_color(color: str) -> str:
        palette = {
            "#1565c0": "#e3f2fd", "#6a1b9a": "#f3e5f5",
            "#1b5e20": "#e8f5e9", "#546e7a": "#eceff1",
            "#37474f": "#f5f5f5",
        }
        return palette.get(color, "#ffffff")

    for y, title, subtitle, color, out_label in STEPS:
        is_source = title == "Foursquare Dataset"
        bstyle = "round,pad=0.1" if is_source else "round,pad=0.08"
        ax.add_patch(mpatches.FancyBboxPatch(
            (CX - BW / 2, y - BH / 2), BW, BH,
            boxstyle=bstyle, linewidth=1.8,
            edgecolor=color, facecolor=box_color(color), zorder=2))
        ax.text(CX, y + 0.14, title,
                ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=color, zorder=3)
        ax.text(CX, y - 0.22, subtitle,
                ha="center", va="center", fontsize=8.5, color="#555555", zorder=3)

        if out_label:
            ax.annotate("", xy=(FX - 0.1, y), xytext=(CX + BW / 2, y),
                        arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.2), zorder=3)
            ax.text(FX + 0.05, y, out_label,
                    ha="left", va="center", fontsize=8, color="#444444",
                    style="italic", zorder=3)

    # Down arrows between steps
    for i in range(len(STEPS) - 1):
        y_from = STEPS[i][0] - BH / 2
        y_to = STEPS[i + 1][0] + BH / 2
        ax.annotate("", xy=(CX, y_to), xytext=(CX, y_from),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8), zorder=3)

    # PostgreSQL DB cylinder-style box at bottom
    db_y = STEPS[-1][0] - BH / 2 - 0.5
    db_h = 1.1
    ax.add_patch(mpatches.FancyBboxPatch(
        (CX - BW / 2, db_y - db_h), BW, db_h,
        boxstyle="round,pad=0.1", linewidth=2.0,
        edgecolor="#1b5e20", facecolor="#c8e6c9", zorder=2))
    ax.add_patch(mpatches.Ellipse(
        (CX, db_y), BW, 0.38,
        linewidth=2.0, edgecolor="#1b5e20", facecolor="#a5d6a7", zorder=3))
    ax.text(CX, db_y - db_h / 2, "PostgreSQL DB",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="#1b5e20", zorder=4)
    ax.text(CX, db_y - db_h / 2 - 0.28, "visits   ·   pois   ·   poi_categories",
            ha="center", va="center", fontsize=9, color="#2e7d32", zorder=4)

    ax.set_title("Pipeline ETL — del CSV Foursquare a PostgreSQL",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(_save("fig_etl_flow.png"), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log("fig_etl_flow — OK")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Bubble Chart — volumen + categorías
# ─────────────────────────────────────────────────────────────────────────────

def fig_bubble_dataset():
    """Bubble chart: 3 ciudades — tamaño ∝ check-ins, donut = distribución categorías."""
    with get_db_connection() as conn:
        stats = _query_city_stats(conn)
        cat_dists = {qid: _query_cat_dist(conn, qid) for qid in CITY_META}

    # Width proportional to sqrt(n_visits) so bubble area ∝ n_visits
    sqrts = {qid: np.sqrt(stats[qid]["n_visits"]) for qid in CITY_META}
    max_sqrt = max(sqrts.values())
    ratios = [sqrts[qid] / max_sqrt for qid in CITY_META]

    fig = plt.figure(figsize=(16, 9), facecolor="#fafafa")
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, figure=fig, width_ratios=ratios, wspace=0.05,
                  left=0.02, right=0.98, top=0.85, bottom=0.18)

    all_cats = sorted(CAT_COLORS.keys())

    for col, (qid, meta) in enumerate(CITY_META.items()):
        ax = fig.add_subplot(gs[0, col])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # Background circle
        bg_r = 1.65
        circle = plt.Circle((0, 0), bg_r, facecolor=meta["color"],
                             alpha=0.10, edgecolor=meta["color"],
                             linewidth=2.5, zorder=0)
        ax.add_patch(circle)

        # Category donut
        dist = cat_dists[qid]
        total = max(sum(dist.values()), 1)
        vals, colors = [], []
        for cat in all_cats:
            v = dist.get(cat, 0)
            if v > 0:
                vals.append(v / total)
                colors.append(CAT_COLORS[cat])

        if vals:
            wedge_props = dict(width=0.52, edgecolor="white", linewidth=1.2)
            ax.pie(vals, colors=colors, wedgeprops=wedge_props,
                   startangle=90, radius=1.25)

        # Center text: city + count
        n_v = stats[qid]["n_visits"]
        ax.text(0, 0.18, meta["name"],
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=meta["color"])
        ax.text(0, -0.22, f"{n_v:,}\ncheck-ins",
                ha="center", va="center", fontsize=9, color="#444444",
                linespacing=1.3)

    # Shared legend at bottom
    legend_items = [
        mpatches.Patch(facecolor=CAT_COLORS[c], label=c)
        for c in all_cats if any(cat_dists[qid].get(c, 0) > 0 for qid in CITY_META)
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=6,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.02), title="Categorías amplias")

    fig.suptitle(
        "Volumen y composición del dataset — 3 ciudades\n"
        "(tamaño de burbuja proporcional a √check-ins)",
        fontsize=13, fontweight="bold", y=0.96,
    )

    fig.savefig(_save("fig_bubble_dataset.png"), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log("fig_bubble_dataset — OK")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Heatmap de cobertura del dataset
# ─────────────────────────────────────────────────────────────────────────────

def fig_heatmap_coverage():
    """Heatmap ciudad × estadísticas del dataset."""
    with get_db_connection() as conn:
        stats = _query_city_stats(conn)

    cities = list(CITY_META.keys())
    city_names = [CITY_META[qid]["name"] for qid in cities]

    METRICS = [
        ("n_users",   "Usuarios únicos"),
        ("n_pois",    "POIs"),
        ("n_visits",  "Check-ins"),
        ("n_trails",  "Trails"),
        ("avg_trail", "Media POIs/trail"),
        ("sparsity",  "Sparsidad matriz"),
    ]

    data_raw = np.array([
        [stats[qid][m] for m, _ in METRICS] for qid in cities
    ], dtype=float)

    # Normalize each column 0-1 for coloring
    col_min = data_raw.min(axis=0)
    col_max = data_raw.max(axis=0)
    col_range = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    data_norm = (data_raw - col_min) / col_range

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#fafafa")

    im = ax.imshow(data_norm, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=1.0)

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([label for _, label in METRICS], fontsize=11,
                       rotation=25, ha="right")
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(city_names, fontsize=12, fontweight="bold")

    # Annotate each cell with actual value
    FMT = ["{:,.0f}", "{:,.0f}", "{:,.0f}", "{:,.0f}", "{:.1f}", "{:.3f}"]
    for i in range(len(cities)):
        for j in range(len(METRICS)):
            val = data_raw[i, j]
            txt = FMT[j].format(val)
            brightness = data_norm[i, j]
            text_color = "white" if brightness > 0.65 else "#212121"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_title("Cobertura del dataset por ciudad",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(top=False, bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Valor normalizado (0–1 por columna)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(_save("fig_heatmap_coverage.png"), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log("fig_heatmap_coverage — OK")


# ─────────────────────────────────────────────────────────────────────────────
# Registry + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _registry() -> dict[str, Callable[[], None]]:
    return {
        "fig_er":       fig_er_diagram,
        "fig_etl":      fig_etl_flow,
        "fig_bubble":   fig_bubble_dataset,
        "fig_heatmap":  fig_heatmap_coverage,
    }


def main():
    parser = argparse.ArgumentParser(description="Figuras de dataset para TFG")
    parser.add_argument("--only", type=str, default="",
                        help="Ejecutar solo una figura, ej: fig_er")
    parser.add_argument("--skip", type=str, default="",
                        help="Saltar figuras (separadas por coma)")
    args = parser.parse_args()

    registry = _registry()
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    names = [args.only] if args.only else list(registry.keys())
    names = [n for n in names if n not in skip]

    ok, fail = [], []
    log(f"Salida: {OUTPUT_DIR.resolve()}")
    for name in names:
        fn = registry.get(name)
        if fn is None:
            log(f"[SKIP] No existe '{name}'")
            continue
        log(f"[RUN] {name}")
        try:
            fn()
            ok.append(name)
            log(f"[OK]  {name}")
        except Exception as e:
            fail.append((name, str(e)))
            log(f"[FAIL] {name}: {e}")
            traceback.print_exc()

    log(f"Finalizado. OK={len(ok)} FAIL={len(fail)}")
    if ok:
        log("OK: " + ", ".join(ok))
    for n, err in fail:
        log(f"- {n}: {err}")


if __name__ == "__main__":
    main()
