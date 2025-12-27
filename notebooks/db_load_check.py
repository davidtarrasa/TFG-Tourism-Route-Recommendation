"""
db_load_check.py: comprobaciones rápidas de la carga en Postgres.

Uso:
  python notebooks/db_load_check.py --dsn postgresql://tfg:tfgpass@localhost:55432/tfg_routes

Si no pasas --dsn usará POSTGRES_DSN del entorno o el valor anterior por defecto.
"""
import argparse
import os

import pandas as pd
import psycopg


def get_connection(dsn: str):
    return psycopg.connect(dsn)


def check_counts(conn):
    q = """
    SELECT
      (SELECT COUNT(*) FROM visits) AS visits,
      (SELECT COUNT(*) FROM pois) AS pois,
      (SELECT COUNT(*) FROM poi_categories) AS poi_categories;
    """
    return pd.read_sql(q, conn)


def city_distribution(conn):
    q = """
    SELECT city, COUNT(*) AS pois
    FROM pois
    GROUP BY city
    ORDER BY pois DESC;
    """
    return pd.read_sql(q, conn)


def top_categories(conn, limit=15):
    q = f"""
    SELECT category_name, COUNT(*) AS cnt
    FROM poi_categories
    GROUP BY category_name
    ORDER BY cnt DESC
    LIMIT {limit};
    """
    return pd.read_sql(q, conn)


def sample_pois(conn, limit=10):
    q = f"""
    SELECT fsq_id, name, city, country, rating, price_tier, primary_category
    FROM pois
    LIMIT {limit};
    """
    return pd.read_sql(q, conn)


def visits_by_city(conn):
    q = """
    SELECT venue_city, COUNT(*) AS visits
    FROM visits
    GROUP BY venue_city
    ORDER BY visits DESC;
    """
    return pd.read_sql(q, conn)


def main():
    parser = argparse.ArgumentParser(description="Checks rápidos sobre la BD cargada")
    parser.add_argument("--dsn", help="postgresql://user:pass@host:port/db")
    args = parser.parse_args()

    dsn = args.dsn or os.getenv("POSTGRES_DSN") or "postgresql://tfg:tfgpass@localhost:55432/tfg_routes"
    with get_connection(dsn) as conn:
        print("== Counts ==")
        print(check_counts(conn))
        print("\n== POIs por ciudad ==")
        print(city_distribution(conn))
        print("\n== Top categorías ==")
        print(top_categories(conn))
        print("\n== Ejemplo de POIs ==")
        print(sample_pois(conn))
        print("\n== Visits por venue_city ==")
        print(visits_by_city(conn))


if __name__ == "__main__":
    main()
