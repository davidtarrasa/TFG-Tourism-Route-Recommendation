"""
routing_geoapify_demo.py: demo de ruteo y visualización.

Uso:
  python notebooks/routing_geoapify_demo.py --city osaka --n 2 --engine geoapify
  (usa GEOAPIFY_API_KEY en .env o variable de entorno; con --engine osrm no gasta cuota)

Salida:
  data/reports/routing_demo_{city}.html con la ruta dibujada.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import psycopg
import requests
import folium
import webbrowser
from dotenv import load_dotenv

load_dotenv()

GEOAPIFY_KEY = os.getenv("GEOAPIFY_API_KEY")
# Carpeta de mapas renderizados
MAPS_DIR = Path("data/reports/maps")
MAPS_DIR.mkdir(parents=True, exist_ok=True)


def get_conn(dsn: str):
    return psycopg.connect(dsn)


def sample_pois(conn, city: str, n=2):
    q = """
    SELECT fsq_id, name, lat, lon, city, primary_category
    FROM pois
    WHERE city ILIKE %(city)s
    AND lat IS NOT NULL AND lon IS NOT NULL
    LIMIT %(n)s;
    """
    return pd.read_sql(q, conn, params={"city": f"%{city}%", "n": n})


def route_geoapify(df: pd.DataFrame, mode="walk"):
    if GEOAPIFY_KEY is None:
        raise SystemExit("No hay GEOAPIFY_API_KEY en entorno/.env")
    if len(df) < 2:
        raise ValueError("Necesitas al menos 2 POIs para trazar una ruta")
    waypoints = "|".join(f"{row.lat},{row.lon}" for _, row in df.iterrows())
    url = (
        "https://api.geoapify.com/v1/routing"
        f"?waypoints={waypoints}&mode={mode}&apiKey={GEOAPIFY_KEY}"
    )
    resp = requests.get(url, timeout=15)
    if not resp.ok:
        raise SystemExit(f"Geoapify error {resp.status_code}: {resp.text}")
    return resp.json()


def route_osrm(df: pd.DataFrame, mode="walk"):
    """
    Usa el demo server de OSRM (gratis). Perfiles típicos: driving, foot, bike.
    """
    if len(df) < 2:
        raise ValueError("Necesitas al menos 2 POIs para trazar una ruta")
    profile = "foot" if mode in ("walk", "foot") else "driving"
    coords = ";".join(f"{row.lon},{row.lat}" for _, row in df.iterrows())
    url = f"http://router.project-osrm.org/route/v1/{profile}/{coords}?overview=full&geometries=geojson"
    resp = requests.get(url, timeout=15)
    if not resp.ok:
        raise SystemExit(f"OSRM error {resp.status_code}: {resp.text}")
    data = resp.json()
    if not data.get("routes"):
        raise SystemExit("OSRM no devolvió rutas")
    # Adaptar al formato GeoJSON esperado por Folium (FeatureCollection)
    geom = data["routes"][0]["geometry"]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"profile": profile, "distance": data['routes'][0].get('distance')},
            }
        ],
    }


def render_map(df: pd.DataFrame, geojson: dict, city: str, out_path: Path, auto_open: bool = False, tiles: str = "cartodbpositron"):
    center = [df["lat"].mean(), df["lon"].mean()]
    # Usa tiles de Geoapify si se pide y hay key
    tile_url = tiles
    tile_attr = None
    if tiles == "geoapify" and GEOAPIFY_KEY:
        tile_url = f"https://maps.geoapify.com/v1/tile/dark-matter/{{z}}/{{x}}/{{y}}.png?apiKey={GEOAPIFY_KEY}"
        tile_attr = "© Geoapify | OpenMapTiles | OpenStreetMap contributors"
    elif tiles == "satellite":
        tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        tile_attr = "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"

    m = folium.Map(location=center, zoom_start=13, tiles=tile_url, attr=tile_attr)

    # Añadir puntos
    for _, row in df.iterrows():
        folium.Marker(
            location=[row.lat, row.lon],
            tooltip=f"{row.name} ({row.primary_category})",
        ).add_to(m)

    # Añadir ruta
    folium.GeoJson(geojson, name="route").add_to(m)
    folium.LayerControl().add_to(m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(out_path)
    print(f"Mapa guardado en {out_path}")
    if auto_open:
        try:
            webbrowser.open(out_path.resolve().as_uri())
            print("Abriendo mapa en el navegador…")
        except Exception as exc:
            print(f"No se pudo abrir automáticamente: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Demo de ruta con Geoapify")
    parser.add_argument("--dsn", default=os.getenv("POSTGRES_DSN") or "postgresql://tfg:tfgpass@localhost:55432/tfg_routes")
    parser.add_argument("--city", default="osaka", help="Nombre de ciudad (coincide con campo city en pois)")
    parser.add_argument("--n", type=int, default=2, help="Número de POIs a tomar (mínimo 2)")
    parser.add_argument("--mode", default="walk", help="Modo de Geoapify (walk, drive, bicycle, etc.)")
    parser.add_argument("--engine", choices=["geoapify", "osrm"], default="geoapify", help="Motor de ruteo")
    parser.add_argument("--open", dest="auto_open", action="store_true", help="Abrir el HTML generado en el navegador")
    parser.add_argument(
        "--tiles",
        default="cartodbpositron",
        help="Tiles para Folium: cartodbpositron (default), stamenterrain, geoapify (requiere key), satellite",
    )
    args = parser.parse_args()

    with get_conn(args.dsn) as conn:
        df = sample_pois(conn, args.city, n=max(args.n, 2))
    print(f"POIs elegidos:\n{df[['name','city','lat','lon','primary_category']]}")

    if args.engine == "geoapify":
        print("Usando Geoapify...")
        route = route_geoapify(df, mode=args.mode)
    else:
        print("Usando OSRM demo (sin API key)...")
        route = route_osrm(df, mode=args.mode)

    out = MAPS_DIR / f"routing_demo_{args.city}.html"
    render_map(df, route, args.city, out, auto_open=args.auto_open, tiles=args.tiles)


if __name__ == "__main__":
    main()
