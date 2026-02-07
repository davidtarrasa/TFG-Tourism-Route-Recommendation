"""Utilidades para construir una ruta ordenada y exportarla (GeoJSON / HTML).

Por defecto:
- Ordena los POIs con una heurística de vecino más cercano + 2-opt ligero sobre distancias Haversine.
- Puede usar un ancla (lat/lon o current_poi) para definir el inicio.
- Puede generar GeoJSON y un mapa Folium.

Nota: la llamada a servicios externos (Geoapify/OSRM) se deja opcional; por ahora usamos Haversine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import folium
import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def nearest_neighbor_order(dist_matrix: np.ndarray, start_idx: int = 0) -> List[int]:
    """
    Nearest-neighbor tour on a precomputed distance matrix.

    Important: do NOT use Euclidean distance on raw lat/lon degrees; use a proper
    metric (we precompute Haversine distances into dist_matrix).
    """
    n = int(dist_matrix.shape[0])
    visited = np.zeros(n, dtype=bool)
    order = [int(start_idx)]
    visited[int(start_idx)] = True
    for _ in range(n - 1):
        last = int(order[-1])
        dists = dist_matrix[last].copy()
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True
    return order


def two_opt(order: List[int], dist_matrix: np.ndarray, max_passes: int = 50, fixed_start: bool = True) -> List[int]:
    """
    Deterministic 2-opt improvement for an *open* path.

    - Uses a "first improvement" strategy (fast and stable for small K).
    - If fixed_start=True, it never changes the first node in the order
      (important when node 0 is a virtual anchor).
    """
    best = [int(x) for x in order]
    n = len(best)
    if n < 4:
        return best

    best_len = float(path_length(best, dist_matrix))
    start_i = 1 if fixed_start else 0

    for _ in range(int(max_passes)):
        improved = False
        for i in range(start_i, n - 2):
            for k in range(i + 1, n - 1):
                new_order = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                new_len = float(path_length(new_order, dist_matrix))
                if new_len + 1e-9 < best_len:
                    best = new_order
                    best_len = new_len
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def path_length(order: Sequence[int], dist_matrix: np.ndarray) -> float:
    return sum(dist_matrix[order[i], order[i + 1]] for i in range(len(order) - 1))


@dataclass
class RouteResult:
    ordered_df: pd.DataFrame
    total_km: float


def build_route(df: pd.DataFrame, anchor_lat: Optional[float] = None, anchor_lon: Optional[float] = None) -> RouteResult:
    """
    Ordena los POIs con heurística NN + 2-opt sobre distancias Haversine.
    Si se pasa anchor_lat/lon, se inserta como primer punto virtual para ordenar y luego se elimina.
    """
    if df.empty:
        return RouteResult(df, 0.0)

    coords = df[["lat", "lon"]].astype(float).to_numpy()
    labels = list(range(len(df)))

    # Insertar ancla virtual
    if anchor_lat is not None and anchor_lon is not None:
        coords = np.vstack([[anchor_lat, anchor_lon], coords])
        labels = [-1] + labels  # -1 representa el ancla

    # Matriz de distancias
    n = coords.shape[0]
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        dist_matrix[i] = haversine_km(coords[i, 0], coords[i, 1], coords[:, 0], coords[:, 1])

    # NN desde el nodo 0 (ancla si existe)
    order = nearest_neighbor_order(dist_matrix, start_idx=0)

    has_anchor = anchor_lat is not None and anchor_lon is not None
    order = two_opt(order, dist_matrix, max_passes=80, fixed_start=has_anchor)

    # Safety: if we have an anchor node (index 0), rotate the path so it starts at it.
    if has_anchor and order and order[0] != 0 and 0 in order:
        pos = order.index(0)
        order = order[pos:] + order[:pos]

    # Si hay ancla, quitarla del orden
    order_labels = [labels[idx] for idx in order if labels[idx] != -1]

    ordered_df = df.iloc[order_labels].reset_index(drop=True)

    # Total length must be computed using the true coordinate indices returned by `order`
    # (including the virtual anchor if present). A remapping approach can produce incorrect totals.
    total_km = path_length(order, dist_matrix) if len(order) > 1 else 0.0
    return RouteResult(ordered_df, float(total_km))


def to_geojson(df: pd.DataFrame) -> dict:
    features = []
    for _, row in df.iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
                "properties": {
                    "fsq_id": row["fsq_id"],
                    "name": row["name"],
                    "primary_category": row.get("primary_category"),
                    "rating": row.get("rating"),
                    "price_tier": row.get("price_tier"),
                    "order": int(row.name) + 1,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def to_folium_map(df: pd.DataFrame, anchor: Optional[Tuple[float, float]] = None) -> folium.Map:
    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())
    # Default to a satellite basemap (more "realistic"). We keep a light basemap as an
    # alternative via LayerControl.
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)

    # Satellite (Esri World Imagery)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    # Light basemap (useful for reading street names)
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Light",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # Ruta (include anchor -> first POI for visual clarity)
    coords = df[["lat", "lon"]].astype(float).to_numpy()
    line = coords.tolist()
    if anchor is not None:
        line = [[float(anchor[0]), float(anchor[1])]] + line
    folium.PolyLine(line, color="blue", weight=4, opacity=0.7).add_to(m)

    # Marcadores
    for idx, row in df.iterrows():
        folium.Marker(
            [float(row["lat"]), float(row["lon"])],
            popup=f"{idx+1}. {row['name']} ({row.get('primary_category','')})",
            tooltip=row["name"],
        ).add_to(m)

    # Anchor
    if anchor is not None:
        folium.Marker(anchor, icon=folium.Icon(color="green", icon="play"), tooltip="Start").add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m
