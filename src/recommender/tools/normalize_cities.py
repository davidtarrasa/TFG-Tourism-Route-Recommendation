"""
Normaliza la columna pois.city a nombres canónicos.

Uso:
  python -m src.recommender.tools.normalize_cities --map osaka

Opciones de mapeo predefinidas:
- osaka: agrupa variantes en japonés a "Osaka".
- full: incluye osaka + algunas variantes comunes en el dataset.
"""

import argparse
from typing import Dict, List

from ..utils_db import get_conn


OSAKA_VARIANTS = [
    "大阪市",
    "大阪市中央区",
    "大阪市西区",
    "大阪市福島区",
    "大阪市天王寺区",
    "大阪市",
    "中央区",
    "港区",
    "北区",
    "浪速区",
    "西成区",
    "梅田",
    "福島区",
    "キタ",
    "都島区",
    "天王寺区",
    "西区",
    "東成区",
    "Osaka",
]

FULL_MAP = {
    "Osaka": OSAKA_VARIANTS,
}


def normalize_city(conn, canonical: str, variants: List[str]):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE pois SET city = %s WHERE city = ANY(%s)",
            (canonical, variants),
        )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Normaliza nombres de ciudad en pois.city")
    parser.add_argument("--map", choices=["osaka", "full"], default="osaka", help="Conjunto de variantes a normalizar")
    parser.add_argument("--dsn", help="DSN de Postgres (si no, usa POSTGRES_DSN o por defecto)")
    args = parser.parse_args()

    conn = get_conn(args.dsn)

    if args.map == "osaka":
        normalize_city(conn, "Osaka", OSAKA_VARIANTS)
        print("Normalizadas variantes de Osaka -> 'Osaka'")
    else:
        for canon, variants in FULL_MAP.items():
            normalize_city(conn, canon, variants)
        print("Normalización completa aplicada")


if __name__ == "__main__":
    main()
