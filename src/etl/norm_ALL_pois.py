# src/etl/build_prof_pois_from_ids.py
# Limpio, organizado y con lista de categorías "gratis"
import os
import json
import argparse
import logging
from collections import Counter
from statistics import median
from typing import Any, Dict, List


# Cargar etiquetas de BERT
BERT_PRICE_PATH = "data/processed/bert_category_price_labels.json"

def load_bert_price_labels():
    if not os.path.exists(BERT_PRICE_PATH):
        return {}
    with open(BERT_PRICE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

BERT_PRICE = load_bert_price_labels()




# =============================================================================
# Configuración / Constantes
# =============================================================================

# Tiers de precio permitidos (incluye 0 = gratis)
ALLOWED_PRICE_TIERS = {0, 1, 2, 3, 4}

# Mapa de ciudades → rutas de entrada/salida
CITY_CFG = {
    # Osaka
    "osaka": {
        "ids":  "data/raw/POIS_osaka_ids.txt",
        "prof": "data/raw/ALL_POIS_Osaka.json",
        "out":  "data/processed/foursquare/ALL_POIS_Osaka_prof_filtered.json",
        "name": "Osaka",
    },
    # Istanbul (alias Estambul)
    "istanbul": {
        "ids":  "data/raw/POIS_istanbul_ids.txt",
        "prof": "data/raw/ALL_POIS_Istanbul.json",
        "out":  "data/processed/foursquare/ALL_POIS_Istanbul_prof_filtered.json",
        "name": "Istanbul",
    },
    "estambul": {
        "ids":  "data/raw/POIS_estambul_ids.txt",
        "prof": "data/raw/ALL_POIS_Istanbul.json",
        "out":  "data/processed/foursquare/ALL_POIS_Istanbul_prof_filtered.json",
        "name": "Istanbul",
    },
    # Petaling Jaya
    "petalingjaya": {
        "ids":  "data/raw/POIS_petalingjaya_ids.txt",
        "prof": "data/raw/ALL_POIS_PetalingJaya.json",
        "out":  "data/processed/foursquare/ALL_POIS_PetalingJaya_prof_filtered.json",
        "name": "Petaling Jaya",
    },
    "petaling jaya": {
        "ids":  "data/raw/POIS_petalingjaya_ids.txt",
        "prof": "data/raw/ALL_POIS_PetalingJaya.json",
        "out":  "data/processed/foursquare/ALL_POIS_PetalingJaya_prof_filtered.json",
        "name": "Petaling Jaya",
    },
}


# =============================================================================
# Utilidades
# =============================================================================

def read_ids(path: str) -> List[str]:
    """Lee IDs, limpia prefijo 'foursquare:' si aparece y deduplica preservando orden."""
    if not os.path.exists(path):
        raise SystemExit(f"No existe el fichero de IDs: {path}")
    with open(path, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    cleaned = [(x.replace("foursquare:", "", 1) if x.startswith("foursquare:") else x) for x in ids]
    seen, out = set(), []
    for x in cleaned:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def load_prof_json(path: str) -> List[dict]:
    """Carga el JSON del profesor (lista o único objeto)."""
    if not os.path.exists(path):
        raise SystemExit(f"No existe el JSON del profesor: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise SystemExit("JSON del profesor con formato inesperado")
    return data

def _extract_lat_lon(obj: dict):
    """Extrae lat/lon desde 'latitude'/'longitude' o 'geocodes.main'."""
    lat = obj.get("latitude")
    lon = obj.get("longitude")
    if lat is None and isinstance(obj.get("geocodes"), dict):
        try:
            lat = obj["geocodes"]["main"]["latitude"]
            lon = obj["geocodes"]["main"]["longitude"]
        except Exception:
            pass
    return lat, lon

def _extract_city_country(obj: dict):
    """Extrae city/country desde 'location'."""
    city = country = None
    loc = obj.get("location")
    if isinstance(loc, dict):
        city = loc.get("locality")
        country = loc.get("country")
    return city, country

def _normalize_categories(cat_list) -> List[Dict[str, Any]]:
    """Devuelve categorías en formato mínimo: [{'id': str, 'name': str}, ...]."""
    out = []
    if isinstance(cat_list, list):
        for c in cat_list:
            if not isinstance(c, dict):
                continue
            cid = c.get("id") or c.get("fsq_category_id")
            name = c.get("name")
            if cid is None or name is None:
                continue
            out.append({"id": str(cid), "name": str(name)})
    return out

def _coerce_float(x):
    """Convierte a float o None."""
    try:
        return float(x)
    except Exception:
        return None

def _coerce_nonneg_int(x):
    """Convierte a entero ≥ 0 o None."""
    try:
        v = int(round(float(x)))
        return v if v >= 0 else 0
    except Exception:
        return None

def _group_median(values):
    """Mediana robusta solo si hay suficientes valores numéricos (≥3)."""
    vals = [v for v in values if isinstance(v, (int, float))]
    if len(vals) >= 3:
        try:
            return float(median(vals))
        except Exception:
            return None
    return None

def _snap_to_allowed_price(x):
    """Ajusta precio a {0,1,2,3,4}; devuelve None si no aplica."""
    if x is None:
        return None
    try:
        r = int(round(float(x)))
    except Exception:
        return None
    if r < 0:
        r = 0
    if r > 4:
        r = 4
    return r if r in ALLOWED_PRICE_TIERS else None

def _fill_missing_city_with_mode(normalized: List[dict]) -> None:
    """Rellena city nulo con la variante más común dentro del propio fichero."""
    ctr = Counter([(p.get("city") or "").strip() for p in normalized if (p.get("city") or "").strip()])
    if not ctr:
        return
    mode_city, _ = ctr.most_common(1)[0]
    for p in normalized:
        if not (p.get("city") or "").strip():
            p["city"] = mode_city

def is_free_category(categories):
    """True si BERT de precio marca la categoría como free."""
    if not categories:
        return False
    for c in categories:
        cid = str(c.get("id"))
        info = BERT_PRICE.get(cid)
        if info and info.get("free") is True:
            return True
    return False


def fill_price_from_bert_price(categories):
    """Devuelve tier 1..4 si BERT de precio lo conoce; None si no hay info."""
    if not categories:
        return None
    for c in categories:
        cid = str(c.get("id"))
        info = BERT_PRICE.get(cid)
        if info and (info.get("free") is False):
            tier = info.get("price_tier")
            if isinstance(tier, int) and 1 <= tier <= 4:
                return tier
    return None

def _to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

def _valid_latlon(lat, lon):
    return (
        isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and
        -90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0
    )

# =============================================================================
# Normalización de objetos
# =============================================================================

def normalize_prof_poi(obj: dict) -> Dict[str, Any]:
    """
    Normaliza un POI del JSON del profesor al esquema común usado en los *_prof_filtered.json.
    Mantiene solo campos necesarios; categorías sin short_name/plural_name.
    """
    if not isinstance(obj, dict):
        return {}

    fsq = obj.get("fsq_id") or obj.get("fsq_place_id")
    if not fsq:
        return {}

    lat, lon = _extract_lat_lon(obj)
    city, country = _extract_city_country(obj)
    cats = _normalize_categories(obj.get("categories"))
    primary_category = cats[0]["name"] if cats else None

    # rating / total_ratings (limpieza de tipos)
    rating = obj.get("rating")
    stats = obj.get("stats")
    total_ratings = stats.get("total_ratings") if isinstance(stats, dict) else None
    if total_ratings is None and isinstance(stats, dict):
        total_ratings = stats.get("usersCount")

    price = obj.get("price")
    if isinstance(price, dict):  # algunos JSON traen {"tier": X}
        price = price.get("tier")

    # Si es "gratis" por categoría, fijamos 0 y marcamos flag
    is_free = False
    if is_free_category(cats):
        price = 0
        is_free = True

    # Tipado final
    rating = _coerce_float(rating)
    total_ratings = _coerce_nonneg_int(total_ratings)
    price = _snap_to_allowed_price(price)

    # Coaccionar lat/lon a float si vienen como string
    lat = _to_float_or_none(lat)
    lon = _to_float_or_none(lon)

    return {
        "fsq_id": str(fsq),
        "lat": lat,
        "lon": lon,
        "city": city,
        "country": country,
        "rating": rating,
        "price": price,
        "total_ratings": total_ratings,
        "categories": cats,
        "primary_category": primary_category,
        "is_free": is_free,  # <- añadido para "saber que son gratis"
    }

# =============================================================================
# Programa principal
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Filtra y normaliza POIs del profesor con los IDs de std_2018"
    )
    parser.add_argument("--city", "-c", required=True, help="Osaka / Istanbul / Estambul / Petaling Jaya")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("build_prof_pois_from_ids")

    # Resolver config por ciudad (admite 'petaling jaya' y 'estambul')
    key = args.city.strip().lower()
    cfg = CITY_CFG.get(key) or CITY_CFG.get(key.replace(" ", ""))
    if not cfg:
        raise SystemExit(f"Ciudad no reconocida. Opciones: {sorted(set(v['name'] for v in CITY_CFG.values()))}")

    ids_path, prof_path, out_path, city_name = cfg["ids"], cfg["prof"], cfg["out"], cfg["name"]

    # Leer IDs (std_2018) y JSON del profesor
    ids = read_ids(ids_path)
    log.info(f"{city_name}: {len(ids):,} IDs leídos de {ids_path}")

    prof = load_prof_json(prof_path)

    # Indexar JSON del profesor por fsq_id/fsq_place_id
    idx: Dict[str, dict] = {}
    for o in prof:
        fid = (o.get("fsq_id") or o.get("fsq_place_id"))
        if fid:
            idx[str(fid)] = o

    # Emparejar por id
    matched_raw = [idx[i] for i in ids if i in idx]
    log.info(f"Coincidencias con profesor: {len(matched_raw):,} / {len(ids):,}")

    # Normalizar y filtrar vacíos
    normalized = [normalize_prof_poi(o) for o in matched_raw]
    normalized = [x for x in normalized if x.get("fsq_id")]

    # Rellenar city nulo con la variante más común del propio fichero
    _fill_missing_city_with_mode(normalized)
    # -------------------------------------------------------------------------
    # Filtro: descartar POIs sin categorías o con lat/lon inválidos
    # -------------------------------------------------------------------------
    before = len(normalized)
    normalized = [
        p for p in normalized
        if (p.get("categories") and _valid_latlon(p.get("lat"), p.get("lon")))
    ]
    removed = before - len(normalized)
    log.info(f"Filtrados {removed} POIs por faltas de categorías o lat/lon inválidos")

    # -------------------------------------------------------------------------
    # Medianas por categoría (usando TODAS las categorías del POI)
    # -------------------------------------------------------------------------
    acc_price, acc_rating, acc_totrat = {}, {}, {}  # cat_id -> list[valor]

    for poi in normalized:
        cats = poi.get("categories") or []
        if not cats:
            continue
        for c in cats:
            cid = c.get("id")
            if not cid:
                continue
            # Solo recolectamos si ya hay valor numérico
            p = poi.get("price")
            if isinstance(p, (int, float)):
                acc_price.setdefault(cid, []).append(p)
            r = poi.get("rating")
            if isinstance(r, (int, float)):
                acc_rating.setdefault(cid, []).append(r)
            tr = poi.get("total_ratings")
            if isinstance(tr, (int, float)):
                acc_totrat.setdefault(cid, []).append(tr)

    by_cat_price, by_cat_rating, by_cat_totrat = {}, {}, {}
    for cid, vals in acc_price.items():
        m = _group_median(vals)
        if m is not None:
            by_cat_price[cid] = _snap_to_allowed_price(m)
    for cid, vals in acc_rating.items():
        m = _group_median(vals)
        if m is not None:
            by_cat_rating[cid] = m
    for cid, vals in acc_totrat.items():
        m = _group_median(vals)
        if m is not None:
            by_cat_totrat[cid] = _coerce_nonneg_int(m)

    # -------------------------------------------------------------------------
    # Imputación por categorías (primaria → secundaria → ...)
    # -------------------------------------------------------------------------
    for poi in normalized:
        cats = poi.get("categories") or []
        cat_ids_in_order = [c.get("id") for c in cats if c.get("id")]

        # --- price ---
        if poi.get("price") in (None, ""):
            tier = fill_price_from_bert_price(poi.get("categories") or [])
            if tier is not None:
                poi["price"] = tier

        # --- rating ---
        if poi.get("rating") in (None, "") or not isinstance(poi.get("rating"), (int, float)):
            for cid in cat_ids_in_order:
                m = by_cat_rating.get(cid)
                if m is not None:
                    poi["rating"] = float(m)
                    break
            # Si no hay mediana para ninguna categoría, se deja None (sin fallback)

        # --- total_ratings ---
        if poi.get("total_ratings") in (None, "") or not isinstance(poi.get("total_ratings"), (int, float)):
            for cid in cat_ids_in_order:
                m = by_cat_totrat.get(cid)
                if m is not None:
                    poi["total_ratings"] = _coerce_nonneg_int(m)
                    break
            # Si no hay mediana para ninguna categoría, se deja 0
            if poi.get("total_ratings") in (None, ""):
                poi["total_ratings"] = 0


    # Guardar
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    log.info(f"Guardado: {out_path}  ({len(normalized):,} POIs normalizados)")

if __name__ == "__main__":
    main()
