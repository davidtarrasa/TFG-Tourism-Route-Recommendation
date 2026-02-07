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


def nearest_neighbor_order(coords: np.ndarray, start_idx: int = 0) -> List[int]:
    n = coords.shape[0]
    visited = np.zeros(n, dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    for _ in range(n - 1):
        last = order[-1]
        dists = np.linalg.norm(coords - coords[last], axis=1)
        dists[visited] = np.inf
        nxt = np.argmin(dists)
        order.append(int(nxt))
        visited[nxt] = True
    return order


def two_opt(order: List[int], dist_matrix: np.ndarray, iterations: int = 200) -> List[int]:
    best = order[:]
    best_len = path_length(best, dist_matrix)
    n = len(order)
    # Need at least 4 nodes to do a meaningful 2-opt swap. With fewer nodes,
    # the random sampling below can fail (n-1 < 2).
    if n < 4:
        return best
    for _ in range(iterations):
        i, k = np.sort(np.random.choice(n - 1, 2, replace=False))
        if i == k:
            continue
        new_order = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
        new_len = path_length(new_order, dist_matrix)
        if new_len < best_len:
            best, best_len = new_order, new_len
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
    order = nearest_neighbor_order(coords, start_idx=0)
    order = two_opt(order, dist_matrix, iterations=min(300, n * 5))

    # Si hay ancla, quitarla del orden
    order_labels = [labels[idx] for idx in order if labels[idx] != -1]

    ordered_df = df.iloc[order_labels].reset_index(drop=True)
    # Recalcular longitud con los índices de coordenadas (sin anchor)
    coord_indices = [i for i, lab in enumerate(labels) if lab != -1]
    coord_map = {lab: coord_idx for coord_idx, lab in zip(coord_indices, order_labels)}
    seq_coords = [coord_map[lab] for lab in order_labels]
    total_km = path_length(seq_coords, dist_matrix) if len(seq_coords) > 1 else 0.0
    return RouteResult(ordered_df, total_km)


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
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

    # Ruta
    coords = df[["lat", "lon"]].astype(float).to_numpy()
    folium.PolyLine(coords, color="blue", weight=4, opacity=0.7).add_to(m)

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

    return m
