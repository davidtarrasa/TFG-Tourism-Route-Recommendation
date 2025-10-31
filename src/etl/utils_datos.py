# src/etl/utils_datos.py
import os
import json
from typing import Dict, Any, List, Optional

import pandas as pd

# -----------------------------------------
# CIUDADES SOPORTADAS (con variantes de nombre)
# -----------------------------------------
CITY_INFO = {
    # Osaka
    "osaka":        {"qid": "wd:Q35765",  "file": "Osaka",        "name": "Osaka",         "country": "Japan"},
    # Istanbul (con alias 'estambul')
    "istanbul":     {"qid": "wd:Q406",    "file": "Istanbul",     "name": "Istanbul",      "country": "Turkey"},
    "estambul":     {"qid": "wd:Q406",    "file": "Istanbul",     "name": "Istanbul",      "country": "Turkey"},
    # Petaling Jaya
    "petalingjaya": {"qid": "wd:Q864965", "file": "PetalingJaya", "name": "Petaling Jaya", "country": "Malaysia"},
    "petaling jaya": {"qid": "wd:Q864965", "file": "PetalingJaya", "name": "Petaling Jaya", "country": "Malaysia"},
}

def get_city_info(city_name: str) -> Dict[str, str]:
    """
    Devuelve info de ciudad: {qid, file, name, country}.
    Normaliza: minúsculas y sin espacios.
    """
    key = city_name.strip().lower()
    if key in CITY_INFO:
        return CITY_INFO[key]
    key = key.replace(" ", "")
    if key in CITY_INFO:
        return CITY_INFO[key]
    raise KeyError(f"Ciudad no reconocida: {city_name}")

# -----------------------------------------
# UTILIDADES BÁSICAS
# -----------------------------------------
def clean_fsq_id(fsq_id: str) -> Optional[str]:
    """Elimina el prefijo 'foursquare:' si existe."""
    if fsq_id is None or pd.isna(fsq_id):
        return None
    s = str(fsq_id)
    return s.replace("foursquare:", "", 1) if s.startswith("foursquare:") else s

def load_std_for_city(std_csv_path: str, city_qid: str) -> pd.DataFrame:
    """
    Carga std_2018.csv, filtra por venue_city == city_qid y devuelve:
    ['trail_id','user_id','timestamp','fsq_id','venue_category','venue_schema'].
    """
    if not os.path.exists(std_csv_path):
        raise FileNotFoundError(f"No se encontró: {std_csv_path}")

    usecols = ["trail_id", "user_id", "venue_id", "venue_category", "venue_schema", "venue_city", "timestamp"]
    df = pd.read_csv(std_csv_path, usecols=usecols, encoding="utf-8")

    df_city = df[df["venue_city"] == city_qid].copy()
    if df_city.empty:
        return df_city

    df_city["fsq_id"] = df_city["venue_id"].map(clean_fsq_id)
    df_city.drop(columns=["venue_id", "venue_city"], inplace=True)
    df_city = df_city.loc[:, ["trail_id", "user_id", "timestamp", "fsq_id", "venue_category", "venue_schema"]]
    return df_city

def load_json_list_safely(path: str) -> List[dict]:
    """Carga un JSON que puede ser lista u objeto. Si no existe, devuelve []."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []

# -----------------------------------------
# NORMALIZADORES DE POIs (profe y API nueva)
# -----------------------------------------
def _first_category_name(cat_list) -> Optional[str]:
    if isinstance(cat_list, list) and cat_list:
        c = cat_list[0]
        if isinstance(c, dict):
            return c.get("name")
    return None

def _extract_lat_lon(obj: dict):
    # API nueva a veces trae 'latitude'/'longitude' arriba del todo
    lat = obj.get("latitude")
    lon = obj.get("longitude")
    # JSON del profesor típico: geocodes.main.latitude/longitude
    if lat is None and isinstance(obj.get("geocodes"), dict):
        try:
            lat = obj["geocodes"]["main"]["latitude"]
            lon = obj["geocodes"]["main"]["longitude"]
        except Exception:
            pass
    return lat, lon

def _extract_city_country(obj: dict):
    city = None
    country = None
    loc = obj.get("location")
    if isinstance(loc, dict):
        city = loc.get("locality")
        country = loc.get("country")
    return city, country

def normalize_poi_obj(obj: dict) -> Dict[str, Any]:
    """
    Normaliza un POI de cualquiera de los dos esquemas:
    - Profesor: fsq_id, geocodes.main.lat/lon, location, price int, rating/stats opcional
    - API actual: fsq_place_id, latitude/longitude, location, price{tier}, rating/ratingSignals
    Devuelve un dict canónico:
      {fsq_id, name, category, lat, lon, city, country, rating, price, total_ratings}
    """
    if not isinstance(obj, dict):
        return {}
    fsq = obj.get("fsq_id") or obj.get("fsq_place_id")
    if not fsq:
        return {}

    name = obj.get("name")
    category = _first_category_name(obj.get("categories"))
    lat, lon = _extract_lat_lon(obj)
    city, country = _extract_city_country(obj)

    rating = obj.get("rating")
    total_ratings = obj.get("ratingSignals")
    if total_ratings is None:
        stats = obj.get("stats")
        if isinstance(stats, dict):
            total_ratings = stats.get("total_ratings") or stats.get("usersCount")

    price = obj.get("price")
    if isinstance(price, dict):  # a veces viene como {"tier": 2}
        price = price.get("tier")

    return {
        "fsq_id": str(fsq),
        "name": name,
        "category": category,
        "lat": lat,
        "lon": lon,
        "city": city,
        "country": country,
        "rating": rating,
        "price": price,
        "total_ratings": total_ratings
    }

def coalesce_record(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Combina dos POIs normalizados del mismo fsq_id; preferencia a valores no nulos de b."""
    out = dict(a)
    for k, v in b.items():
        if k == "fsq_id":
            out["fsq_id"] = a.get("fsq_id") or b.get("fsq_id")
        else:
            out[k] = v if v not in (None, "", []) else out.get(k)
    return out

def coalesce_pois(prof_norm: List[Dict[str, Any]],
                  api_norm: List[Dict[str, Any]],
                  prefer: str = "api") -> Dict[str, Dict[str, Any]]:
    """
    Une listas de POIs normalizados por fsq_id.
    prefer='api' → los valores de API pisan a los del profe si no son nulos.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for p in prof_norm:
        fid = p.get("fsq_id")
        if not fid:
            continue
        merged[fid] = p
    for q in api_norm:
        fid = q.get("fsq_id")
        if not fid:
            continue
        if fid in merged:
            merged[fid] = coalesce_record(merged[fid], q) if prefer == "api" else coalesce_record(q, merged[fid])
        else:
            merged[fid] = q
    return merged

# -----------------------------------------
# CARGA DE POIs EN FORMA DICCIONARIO (para integracion)
# -----------------------------------------
def load_pois_data(pois_json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Carga POIs desde JSON (profe, API o merged), normaliza cada entrada y
    devuelve dict {fsq_id: poi_normalizado}.
    """
    data = load_json_list_safely(pois_json_path)
    pois_dict: Dict[str, Dict[str, Any]] = {}
    for obj in data:
        norm = normalize_poi_obj(obj)
        fid = norm.get("fsq_id")
        if not fid:
            continue
        # Si hay duplicados, nos quedamos con el primero que tenga más campos completos
        if fid in pois_dict:
            # preferimos el que tenga más valores no nulos
            cur = pois_dict[fid]
            n_cur = sum(v not in (None, "", []) for v in cur.values())
            n_new = sum(v not in (None, "", []) for v in norm.values())
            if n_new > n_cur:
                pois_dict[fid] = norm
        else:
            pois_dict[fid] = norm
    return pois_dict
