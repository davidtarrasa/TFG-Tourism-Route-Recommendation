"""Utilidades comunes para el pipeline ETL (src2_prueba/utils.py)."""
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

# Ciudades soportadas (QID, nombre de fichero base y nombre legible)
CITY_INFO: Dict[str, Dict[str, str]] = {
    "osaka": {
        "qid": "Q35765",
        "file": "Osaka",
        "name": "Osaka",
        "country": "Japan",
    },
    "istanbul": {
        "qid": "Q406",
        "file": "Istanbul",
        "name": "Istanbul",
        "country": "Turkey",
    },
    "estambul": {
        "qid": "Q406",
        "file": "Istanbul",
        "name": "Istanbul",
        "country": "Turkey",
    },
    "petalingjaya": {
        "qid": "Q864965",
        "file": "PetalingJaya",
        "name": "Petaling Jaya",
        "country": "Malaysia",
    },
    "petaling jaya": {
        "qid": "Q864965",
        "file": "PetalingJaya",
        "name": "Petaling Jaya",
        "country": "Malaysia",
    },
}


def get_city_config(city: str) -> Dict[str, str]:
    """Normaliza la ciudad y devuelve su configuración."""
    key = city.strip().lower()
    cfg = CITY_INFO.get(key) or CITY_INFO.get(key.replace(" ", ""))
    if not cfg:
        raise KeyError(f"Ciudad no reconocida: {city}")
    return cfg


def clean_prefix(val: str, prefix: str) -> str:
    """Elimina un prefijo si existe."""
    if isinstance(val, str) and val.startswith(prefix):
        return val.replace(prefix, "", 1)
    return val


def clean_fsq_id(fsq_id: str) -> Optional[str]:
    """Normaliza IDs de Foursquare quitando el prefijo estándar."""
    if fsq_id is None or pd.isna(fsq_id):
        return None
    s = str(fsq_id).strip()
    return s.replace("foursquare:", "", 1) if s.startswith("foursquare:") else s


def load_json_list(path: str) -> List[dict]:
    """Carga un JSON que puede ser lista u objeto. Devuelve siempre lista de dicts."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def save_json(obj: Any, path: str):
    """Guarda un objeto JSON con ensure_ascii=False e indentado."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_std(std_path: str) -> pd.DataFrame:
    """Lee el CSV std con columnas mínimas."""
    usecols = [
        "trail_id",
        "user_id",
        "venue_id",
        "venue_category",
        "venue_schema",
        "venue_city",
        "venue_country",
        "timestamp",
    ]
    return pd.read_csv(std_path, usecols=usecols, encoding="utf-8")


def normalize_poi_obj(obj: dict) -> Dict[str, Any]:
    """Normaliza un POI al esquema canónico empleado en el pipeline."""
    if not isinstance(obj, dict):
        return {}
    fsq = obj.get("fsq_id") or obj.get("fsq_place_id")
    if not fsq:
        return {}

    lat = obj.get("latitude")
    lon = obj.get("longitude")
    if lat is None and isinstance(obj.get("geocodes"), dict):
        try:
            lat = obj["geocodes"]["main"]["latitude"]
            lon = obj["geocodes"]["main"]["longitude"]
        except Exception:
            pass

    city = country = None
    loc = obj.get("location")
    if isinstance(loc, dict):
        city = loc.get("locality")
        country = loc.get("country")

    categories = []
    raw_cats = obj.get("categories") if isinstance(obj.get("categories"), list) else []
    for c in raw_cats:
        if not isinstance(c, dict):
            continue
        cid = c.get("id") or c.get("fsq_category_id")
        name = c.get("name")
        if cid and name:
            categories.append({"id": str(cid), "name": name})
    primary_cat = categories[0]["name"] if categories else None

    rating = obj.get("rating")
    total_ratings = None
    stats = obj.get("stats")
    if isinstance(stats, dict):
        total_ratings = stats.get("total_ratings") or stats.get("usersCount")
    if total_ratings is None:
        total_ratings = obj.get("ratingSignals")

    price = obj.get("price")
    if isinstance(price, dict):
        price = price.get("tier")

    name = obj.get("name") or primary_cat or f"POI-{fsq}"

    return {
        "fsq_id": str(fsq),
        "name": name,
        "lat": lat,
        "lon": lon,
        "city": city,
        "country": country,
        "rating": rating,
        "total_ratings": total_ratings,
        "price": price,
        "categories": categories,
        "primary_category": primary_cat,
    }


def merge_pois(prof_list: List[Dict[str, Any]], api_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Une listas de POIs por fsq_id, priorizando valores no vacíos de la API."""
    merged: Dict[str, Dict[str, Any]] = {}
    for p in prof_list:
        fid = p.get("fsq_id")
        if fid:
            merged[fid] = p
    for q in api_list:
        fid = q.get("fsq_id")
        if not fid:
            continue
        if fid in merged:
            base = merged[fid]
            for k, v in q.items():
                if k == "fsq_id":
                    continue
                if v not in (None, "", []):
                    base[k] = v
            merged[fid] = base
        else:
            merged[fid] = q
    return merged


def ensure_latlon(lat, lon) -> bool:
    """Comprueba coordenadas válidas."""
    try:
        return (-90 <= float(lat) <= 90) and (-180 <= float(lon) <= 180)
    except Exception:
        return False


def safe_int(x):
    """Convierte a int de forma segura."""
    try:
        return int(round(float(x)))
    except Exception:
        return None
