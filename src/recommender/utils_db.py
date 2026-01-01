"""
Utilidades para conexión y consulta a Postgres.
- Obtener DSN desde entorno (.env).
- Helpers para cargar visits/pois/poi_categories en DataFrames.
"""

import os

import pandas as pd
import psycopg


def get_conn(dsn: str | None = None):
    dsn = dsn or os.getenv("POSTGRES_DSN") or "postgresql://tfg:tfgpass@localhost:55432/tfg_routes"
    return psycopg.connect(dsn)


def load_pois(conn) -> pd.DataFrame:
    # TODO: ajustar columnas necesarias.
    return pd.read_sql("SELECT * FROM pois", conn)


def load_visits(conn) -> pd.DataFrame:
    # TODO: ajustar columnas necesarias.
    return pd.read_sql("SELECT * FROM visits", conn)
