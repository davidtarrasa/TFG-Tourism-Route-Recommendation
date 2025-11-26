"""08_load_postgres.py: carga std_clean y POIs enriquecidos en Postgres."""
import argparse
import os
from pathlib import Path

import pandas as pd
import psycopg

from utils import get_city_config, load_json_list

SCHEMA_PATH = Path("sql/schema.sql")
STD_CLEAN = Path("data/processed/std_clean.csv")


def load_schema(conn, schema_path: Path):
    if schema_path.exists():
        sql = schema_path.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print(f"[8/8] Esquema aplicado desde {schema_path}")
    else:
        print(f"[8/8] Aviso: no se encontró {schema_path}, se asume esquema ya creado")


def copy_csv(conn, csv_path: Path, table: str):
    with conn.cursor() as cur, csv_path.open("r", encoding="utf-8") as f:
        cur.copy(f"COPY {table} FROM STDIN WITH CSV HEADER", f)
    conn.commit()
    print(f"[8/8] Cargado CSV -> {table}")


def insert_pois(conn, pois: list):
    rows = [
        (
            p.get("fsq_id"),
            p.get("name"),
            p.get("lat"),
            p.get("lon"),
            p.get("city"),
            p.get("country"),
            p.get("rating"),
            p.get("price"),
            p.get("total_ratings"),
            p.get("primary_category"),
            p.get("is_free", False),
        )
        for p in pois
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO pois (fsq_id, name, lat, lon, city, country, rating, price_tier, total_ratings, primary_category, is_free)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (fsq_id) DO UPDATE
            SET name=EXCLUDED.name, lat=EXCLUDED.lat, lon=EXCLUDED.lon,
                city=EXCLUDED.city, country=EXCLUDED.country, rating=EXCLUDED.rating,
                price_tier=EXCLUDED.price_tier, total_ratings=EXCLUDED.total_ratings,
                primary_category=EXCLUDED.primary_category, is_free=EXCLUDED.is_free
            """,
            rows,
        )
    conn.commit()
    print(f"[8/8] Insertados/actualizados {len(rows):,} POIs")


def insert_poi_categories(conn, pois: list):
    cat_rows = []
    for p in pois:
        fid = p.get("fsq_id")
        cats = p.get("categories") or []
        for c in cats:
            cid = c.get("id")
            cname = c.get("name")
            if cid and cname:
                cat_rows.append((fid, str(cid), cname))
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO poi_categories (fsq_id, category_id, category_name)
            VALUES (%s,%s,%s)
            ON CONFLICT DO NOTHING
            """,
            cat_rows,
        )
    conn.commit()
    print(f"[8/8] Insertadas {len(cat_rows):,} filas en poi_categories")


def main():
    parser = argparse.ArgumentParser(description="Carga datos en Postgres")
    parser.add_argument("--dsn", help="postgresql://user:pass@host:port/db")
    parser.add_argument("--schema", default=str(SCHEMA_PATH), help="Ruta a schema.sql")
    parser.add_argument("--pois-pattern", dest="pois_pattern", default="data/processed/pois_enriched_{name}.json")
    args = parser.parse_args()

    dsn = args.dsn or os.getenv("POSTGRES_DSN") or "postgresql://postgres:postgres@localhost:5432/postgres"
    with psycopg.connect(dsn) as conn:
        load_schema(conn, Path(args.schema))
        if STD_CLEAN.exists():
            copy_csv(conn, STD_CLEAN, "visits")
        else:
            print(f"[8/8] Aviso: no se encontró {STD_CLEAN}")

        for city in ["osaka", "istanbul", "petalingjaya"]:
            cfg = get_city_config(city)
            pois_path = Path(args.pois_pattern.format(name=cfg["file"]))
            if not pois_path.exists():
                print(f"[8/8] Aviso: no se encontró {pois_path}, salto ciudad")
                continue
            pois = load_json_list(str(pois_path))
            insert_pois(conn, pois)
            insert_poi_categories(conn, pois)

    print("[8/8] Carga en Postgres finalizada")


if __name__ == "__main__":
    main()
