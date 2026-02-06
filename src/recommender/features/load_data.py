"""
Carga unificada de datos desde Postgres para los modelos de recomendación.
- Visits: trail_id, user_id, venue_id, venue_city, venue_country, timestamp.
- POIs: fsq_id, coords, ciudad, rating, precio, categoría primaria, is_free.
- Poi_categories: lista de categorías por POI.
"""

from typing import Optional, Tuple

import pandas as pd

from ..utils_db import (
    DEFAULT_DSN,
    get_conn,
    load_poi_categories,
    load_pois,
    load_visits,
)


def load_all(
    dsn: Optional[str] = None,
    city: Optional[str] = None,
    city_qid: Optional[str] = None,
    visits_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (visits_df, pois_df, poi_cats_df).

    Args:
        dsn: cadena DSN de Postgres; si no se pasa, usa POSTGRES_DSN o DEFAULT_DSN.
        city: filtro opcional sobre pois.city (nombre visible).
        city_qid: filtro opcional sobre visits.venue_city (QID).
        visits_limit: limitar filas de visits (para pruebas).
    """
    # Use DSN if provided; otherwise get_conn() falls back to POSTGRES_DSN/DEFAULT_DSN.
    with get_conn(dsn) as conn:
        visits_df = load_visits(conn, city_qid=city_qid, limit=visits_limit)
        pois_df = load_pois(conn, city=city)

        # If the user filters by city_qid (visits) but not by city name (pois),
        # restrict the candidate POIs to those that appear in the visits for that city.
        # This avoids cross-city leakage in cold-start / popularity fallback.
        if city_qid and not city and not visits_df.empty:
            venue_ids = visits_df["venue_id"].astype(str).dropna().unique().tolist()
            pois_df = pois_df[pois_df["fsq_id"].astype(str).isin(venue_ids)].copy()

        poi_cats_df = load_poi_categories(conn, fsq_ids=pois_df["fsq_id"])
    return visits_df, pois_df, poi_cats_df


__all__ = ["load_all"]
