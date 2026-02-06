"""
Utilidades para conexión y consulta a Postgres.
- Obtiene DSN desde entorno (.env).
- Helpers para cargar visits, pois y poi_categories en DataFrames.
"""

import os
from typing import Iterable, Optional

import pandas as pd
import psycopg

DEFAULT_DSN = "postgresql://tfg:tfgpass@localhost:55432/tfg_routes"


def get_conn(dsn: Optional[str] = None):
    """Devuelve una conexión psycopg al DSN indicado o al definido en POSTGRES_DSN."""
    dsn = dsn or os.getenv("POSTGRES_DSN") or DEFAULT_DSN
    return psycopg.connect(dsn)


def load_pois(
    conn,
    city: Optional[str] = None,
    city_qid: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Carga POIs; opcionalmente filtra por city (pois.city) y/o city_qid (pois.city_qid)."""
    cols = (
        columns
        or [
            "fsq_id",
            "name",
            "lat",
            "lon",
            "city",
            "city_qid",
            "country",
            "rating",
            "price_tier",
            "total_ratings",
            "primary_category",
            "is_free",
        ]
    )
    base = f"SELECT {', '.join(cols)} FROM pois"
    clauses = []
    params = {}
    if city:
        clauses.append("city ILIKE %(city)s")
        params["city"] = city
    if city_qid:
        clauses.append("city_qid = %(city_qid)s")
        params["city_qid"] = city_qid
    if clauses:
        return pd.read_sql(f"{base} WHERE " + " AND ".join(clauses), conn, params=params)
    return pd.read_sql(base, conn, params=None)


def load_poi_categories(conn, fsq_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Carga poi_categories; opcionalmente filtra por un conjunto de fsq_id."""
    query = "SELECT fsq_id, category_id, category_name FROM poi_categories"
    if fsq_ids is not None:
        ids_list = list(fsq_ids)
        if ids_list:
            return pd.read_sql(
                query + " WHERE fsq_id = ANY(%(ids)s)",
                conn,
                params={"ids": ids_list},
            )
    return pd.read_sql(query, conn)


def load_visits(conn, city_qid: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """Carga visits; opcionalmente filtra por venue_city (QID) y limita filas."""
    query = "SELECT trail_id, user_id, venue_id, venue_city, venue_country, timestamp FROM visits"
    clauses = []
    params = {}
    if city_qid:
        clauses.append("venue_city = %(city_qid)s")
        params["city_qid"] = city_qid
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    if limit:
        query += " LIMIT %(limit)s"
        params["limit"] = limit
    return pd.read_sql(query, conn, params=params or None)


__all__ = ["get_conn", "load_pois", "load_poi_categories", "load_visits", "DEFAULT_DSN"]
