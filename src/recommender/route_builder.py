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
import colorsys
import json
import os
import html
from urllib import parse, request

import folium
from folium.plugins import PolyLineTextPath
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


def _normalize_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode in ("walk", "walking", "foot"):
        return "walk"
    if mode in ("drive", "driving", "car"):
        return "drive"
    return mode


def _geoapify_mode(mode: str) -> str:
    mode = _normalize_mode(mode)
    if mode == "walk":
        return "walk"
    if mode == "drive":
        return "drive"
    return "drive"


def _route_color(mode: str) -> str:
    mode = _normalize_mode(mode)
    if mode == "walk":
        return "#1f77b4"
    if mode == "drive":
        return "#ff3d00"
    return "#0088ff"


def _route_label(mode: str) -> str:
    mode = _normalize_mode(mode)
    if mode == "walk":
        return "Walking route"
    if mode == "drive":
        return "Driving route"
    return f"Route ({mode})"


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _leg_palette(mode: str, n_legs: int) -> List[str]:
    mode = _normalize_mode(mode)
    n = max(int(n_legs), 1)
    if mode == "drive":
        # Blue cascading palette with enough contrast and no repeats.
        h0, h1 = 225.0 / 360.0, 190.0 / 360.0
        return [_hsv_to_hex(h0 + (h1 - h0) * (i / max(n - 1, 1)), 0.85, 0.96 - 0.14 * (i / max(n - 1, 1))) for i in range(n)]
    if mode == "walk":
        # Cool cascading palette (blue -> cyan -> green) with no repeats.
        h0, h1 = 215.0 / 360.0, 145.0 / 360.0
        return [_hsv_to_hex(h0 + (h1 - h0) * (i / max(n - 1, 1)), 0.78, 0.93 - 0.10 * (i / max(n - 1, 1))) for i in range(n)]
    return [_hsv_to_hex(205.0 / 360.0, 0.72, 0.92) for _ in range(n)]


def _coords_for_route(df: pd.DataFrame, anchor: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
    coords = [(float(r["lat"]), float(r["lon"])) for _, r in df.iterrows()]
    if anchor is not None:
        coords = [(float(anchor[0]), float(anchor[1]))] + coords
    return coords


def _node_labels(df: pd.DataFrame, anchor: Optional[Tuple[float, float]]) -> List[str]:
    labels = [f"{i+1}. {str(row.get('name', 'POI'))}" for i, (_, row) in enumerate(df.iterrows())]
    if anchor is not None:
        return ["Start"] + labels
    return labels


def _geoapify_leg(
    start: Tuple[float, float],
    end: Tuple[float, float],
    mode: str,
    api_key: str,
    timeout_s: float = 8.0,
) -> Optional[List[Tuple[float, float]]]:
    m = _geoapify_mode(mode)
    # Geoapify routing expects "lon,lat|lon,lat"
    waypoints = f"{start[1]},{start[0]}|{end[1]},{end[0]}"
    params = parse.urlencode({"waypoints": waypoints, "mode": m, "apiKey": api_key})
    url = f"https://api.geoapify.com/v1/routing?{params}"
    try:
        with request.urlopen(url, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        features = payload.get("features") or []
        if not features:
            return None
        geometry = features[0].get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if not coords:
            return None
        # GeoJSON comes as [lon, lat] -> folium expects [lat, lon]
        out = [(float(c[1]), float(c[0])) for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        return out if out else None
    except Exception:
        return None


def _compose_real_path(
    points: List[Tuple[float, float]],
    mode: str,
    api_key: Optional[str],
) -> List[Tuple[float, float]]:
    if len(points) < 2:
        return points
    if not api_key:
        return points
    full: List[Tuple[float, float]] = []
    for i in range(len(points) - 1):
        leg = _geoapify_leg(points[i], points[i + 1], mode=mode, api_key=api_key)
        if not leg:
            leg = [points[i], points[i + 1]]
        if not full:
            full.extend(leg)
        else:
            # Avoid duplicate junction point between legs.
            full.extend(leg[1:])
    return full


def _osrm_profile(mode: str) -> str:
    mode = _normalize_mode(mode)
    if mode == "walk":
        return "foot"
    return "driving"


def _osrm_leg(
    start: Tuple[float, float],
    end: Tuple[float, float],
    mode: str,
    timeout_s: float = 8.0,
) -> Optional[List[Tuple[float, float]]]:
    profile = _osrm_profile(mode)
    # OSRM expects lon,lat
    coord = f"{start[1]},{start[0]};{end[1]},{end[0]}"
    params = parse.urlencode({"overview": "full", "geometries": "geojson"})
    url = f"https://router.project-osrm.org/route/v1/{profile}/{coord}?{params}"
    try:
        with request.urlopen(url, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        routes = payload.get("routes") or []
        if not routes:
            return None
        geometry = routes[0].get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if not coords:
            return None
        out = [(float(c[1]), float(c[0])) for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        return out if out else None
    except Exception:
        return None


def _route_leg(
    start: Tuple[float, float],
    end: Tuple[float, float],
    mode: str,
    api_key: Optional[str],
) -> List[Tuple[float, float]]:
    leg = None
    if api_key:
        leg = _geoapify_leg(start, end, mode=mode, api_key=api_key)
    if not leg:
        leg = _osrm_leg(start, end, mode=mode)
    if not leg:
        leg = [start, end]
    return leg


def _load_api_key_from_env_file() -> Optional[str]:
    try:
        env_path = os.path.join(os.getcwd(), ".env")
        if not os.path.exists(env_path):
            return None
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                if key.strip() != "GEOAPIFY_API_KEY":
                    continue
                # Support inline comments in .env
                value = val.split("#", 1)[0].strip().strip('"').strip("'")
                return value or None
    except Exception:
        return None
    return None


def to_folium_map(
    df: pd.DataFrame,
    anchor: Optional[Tuple[float, float]] = None,
    route_modes: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
) -> folium.Map:
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

    route_modes = route_modes or ("walk", "drive")
    clean_modes = []
    for mode in route_modes:
        nm = _normalize_mode(mode)
        if nm not in clean_modes:
            clean_modes.append(nm)
    points = _coords_for_route(df, anchor=anchor)
    labels = _node_labels(df, anchor=anchor)
    key = api_key if api_key is not None else os.getenv("GEOAPIFY_API_KEY") or _load_api_key_from_env_file()

    # Always show straight edges as a reference layer.
    straight_grp = folium.FeatureGroup(name="Straight edges", show=True)
    # Warm dashed edges with very small variation (orange/red family).
    straight_colors = ["#ff6d00", "#ff7043", "#ff5722", "#f4511e"]
    for i in range(len(points) - 1):
        leg = [points[i], points[i + 1]]
        color = straight_colors[i % len(straight_colors)]
        line = folium.PolyLine(
            leg,
            color=color,
            weight=3,
            opacity=0.95,
            dash_array="10,6",
            tooltip=f"Straight edge {i+1}: {i+1} -> {i+2}",
        )
        line.add_to(straight_grp)
    straight_grp.add_to(m)

    # Draw one layer per travel mode (walking / driving), using real roads when possible.
    legend_sections: List[str] = []
    leg_layers: List[Tuple[str, str]] = []
    for idx, mode in enumerate(clean_modes):
        grp = folium.FeatureGroup(name=_route_label(mode), show=(idx == 0))
        palette = _leg_palette(mode, len(points) - 1)
        leg_items: List[str] = []
        for i in range(len(points) - 1):
            leg = _route_leg(points[i], points[i + 1], mode=mode, api_key=key)
            color = palette[i % len(palette)]
            from_label = html.escape(labels[i]) if i < len(labels) else f"P{i+1}"
            to_label = html.escape(labels[i + 1]) if i + 1 < len(labels) else f"P{i+2}"
            leg_title = f"{_route_label(mode)} leg {i+1}: {from_label} -> {to_label}"
            leg_group = folium.FeatureGroup(name=f"{_route_label(mode)} segment {i+1}", show=True, control=False)
            line = folium.PolyLine(
                leg,
                color=color,
                weight=4 if mode == "drive" else 4.5,
                opacity=0.98,
                tooltip=leg_title,
            )
            line.add_to(leg_group)
            try:
                PolyLineTextPath(
                    line,
                    " > ",
                    repeat=True,
                    offset=7,
                    attributes={"fill": color, "font-weight": "bold", "font-size": "14"},
                ).add_to(leg_group)
            except Exception:
                pass
            leg_group.add_to(grp)
            layer_name = leg_group.get_name()
            leg_layers.append((layer_name, f"{_route_label(mode)} - Tramo {i+1}"))
            leg_items.append(
                f"<div style='display:flex;align-items:flex-start;gap:8px;margin:4px 0;'>"
                f"<span style='display:inline-block;flex:0 0 16px;width:16px;height:16px;border-radius:3px;background:{color};border:2px solid #111;margin-top:2px;'></span>"
                f"<label style='display:flex;align-items:flex-start;gap:6px;cursor:pointer;line-height:1.25;'>"
                f"<input type='checkbox' checked "
                f"data-layer='{layer_name}' "
                f"onchange='toggleRouteLeg(this.dataset.layer, this.checked)'/>"
                f"<span style='display:inline-block;'>Tramo {i+1}: {from_label} -> {to_label}</span>"
                f"</label>"
                f"</div>"
            )
        grp.add_to(m)
        legend_sections.append(
            "<div style='margin-top:6px;'>"
            f"<div style='font-weight:700;margin-bottom:4px;'>{html.escape(_route_label(mode))}</div>"
            + "".join(leg_items)
            + "</div>"
        )

    # Small numbered badges over markers to make route order visually explicit.
    badge_css = """
    <style>
      .route-order-badge {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #0b3d91;
        color: #fff;
        font-size: 11px;
        font-weight: 800;
        line-height: 20px;
        text-align: center;
        border: 2px solid #ffffff;
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.45);
      }
    </style>
    """
    m.get_root().html.add_child(folium.Element(badge_css))

    # Marcadores
    for idx, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f"<div class='route-order-badge'>{idx+1}</div>",
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                class_name="route-order-icon",
            ),
            popup=f"{idx+1}. {row['name']} ({row.get('primary_category','')})",
            tooltip=f"Orden {idx+1}",
        ).add_to(m)

    # Anchor
    if anchor is not None:
        folium.Marker(anchor, icon=folium.Icon(color="green", icon="play"), tooltip="Start").add_to(m)

    # On-map legend with segment color order.
    legend_html = (
        "<div style=\"position: fixed; bottom: 20px; right: 20px; z-index: 9999; "
        "background: rgba(255,255,255,0.95); border: 2px solid #333; border-radius: 8px; "
        "padding: 10px 12px; min-width: 300px; max-width: 420px; max-height: 45vh; overflow:auto; "
        "font-family: Arial, sans-serif; font-size: 12px; line-height: 1.35; color:#111;\">"
        "<div style='font-size:13px;font-weight:800;margin-bottom:6px;'>Orden de tramos (color)</div>"
        + "".join(legend_sections)
        + "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    map_js_name = m.get_name()
    leg_registry = ", ".join([f"'{lname}': '{html.escape(title)}'" for lname, title in leg_layers])
    legend_js = f"""
    <script>
      window.__routeLegRegistry = {{{leg_registry}}};
      window.toggleRouteLeg = function(layerName, checked) {{
        try {{
          var mapObj = {map_js_name};
          var layer = window[layerName];
          if (!mapObj || !layer) return;
          if (checked) {{
            if (!mapObj.hasLayer(layer)) mapObj.addLayer(layer);
          }} else {{
            if (mapObj.hasLayer(layer)) mapObj.removeLayer(layer);
          }}
        }} catch (e) {{
          console.warn('toggleRouteLeg failed:', e);
        }}
      }};
    </script>
    """
    m.get_root().html.add_child(folium.Element(legend_js))

    folium.LayerControl(collapsed=True).add_to(m)
    return m
