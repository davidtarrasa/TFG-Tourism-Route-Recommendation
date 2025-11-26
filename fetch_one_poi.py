"""Script de prueba: obtiene un único POI de Foursquare por fsq_id y lo muestra en pantalla.
Uso: python fetch_one_poi.py --id <fsq_id>
Requiere FOURSQUARE_API_KEY en .env o entorno.
"""
import argparse
import json
import os

import requests
from dotenv import load_dotenv

API_BASE = "https://places-api.foursquare.com/places/{}"
HEADERS = {
    "accept": "application/json",
    "X-Places-Api-Version": "2025-06-17",
}


def fetch_one(fsq_id: str, api_key: str):
    url = API_BASE.format(fsq_id)
    headers = dict(HEADERS)
    headers["authorization"] = f"Bearer {api_key}"
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code == 200:
        data = r.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"HTTP {r.status_code}: {r.text[:200]}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Test: recupera un POI por fsq_id desde la API")
    parser.add_argument("--id", required=True, help="fsq_id a consultar")
    args = parser.parse_args()

    api_key = os.getenv("FOURSQUARE_API_KEY")
    if not api_key:
        raise SystemExit("FOURSQUARE_API_KEY no definida en entorno o .env")

    fetch_one(args.id, api_key)


if __name__ == "__main__":
    main()
