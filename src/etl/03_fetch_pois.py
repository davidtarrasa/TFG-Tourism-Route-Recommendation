# -*- coding: utf-8 -*-
"""03_fetch_pois.py: descarga POIs por IDs desde Foursquare (async, reanudable)."""
import argparse
import asyncio
import json
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import httpx
import pandas as pd
from dotenv import load_dotenv

from utils import get_city_config, clean_fsq_id, save_json

API_BASE = "https://places-api.foursquare.com/places/{}"
STD_PATH = "data/processed/std_clean.csv"


def read_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        ids = [clean_fsq_id(ln.strip()) for ln in f if ln.strip()]
    seen, out = set(), []
    for x in ids:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_existing(out_path: str) -> Tuple[List[dict], Set[str]]:
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return [], set()
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            ids = {obj.get("fsq_id") for obj in data if isinstance(obj, dict)}
            return data, {str(i) for i in ids if i}
    except Exception:
        pass
    return [], set()


class RateLimiter:
    def __init__(self, max_per_second: Optional[int]):
        self.max_per_second = max_per_second
        self.q = deque()
        self.lock = asyncio.Lock()

    async def wait(self):
        if not self.max_per_second or self.max_per_second <= 0:
            return
        async with self.lock:
            now = asyncio.get_event_loop().time()
            while self.q and (now - self.q[0]) > 1.0:
                self.q.popleft()
            while len(self.q) >= self.max_per_second:
                await asyncio.sleep(0.01)
                now = asyncio.get_event_loop().time()
                while self.q and (now - self.q[0]) > 1.0:
                    self.q.popleft()
            self.q.append(asyncio.get_event_loop().time())


async def fetch_one(client: httpx.AsyncClient, fsq_id: str, limiter: RateLimiter, max_retries: int = 5):
    url = API_BASE.format(fsq_id)
    for attempt in range(max_retries):
        await limiter.wait()
        try:
            r = await client.get(url, timeout=20.0)
            if r.status_code == 200:
                obj = r.json()
                if isinstance(obj, dict):
                    obj.setdefault("fsq_id", fsq_id)
                return obj
            if r.status_code in (429, 500, 503):
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (1.5 + attempt)
                await asyncio.sleep(wait)
                continue
            logging.warning(f"[{fsq_id}] HTTP {r.status_code}: {r.text[:120]}")
            return None
        except httpx.RequestError as e:
            logging.warning(f"[{fsq_id}] Error de red: {e}. Reintentando...")
            await asyncio.sleep(1.0 + attempt * 0.5)
    return None


def combine_and_save(out_path: str, existing: List[dict], new_data: Dict[str, dict]) -> int:
    combined: Dict[str, dict] = {}
    for obj in existing:
        fid = obj.get("fsq_id")
        if isinstance(fid, str) and fid not in combined:
            combined[fid] = obj
    for fid, obj in new_data.items():
        if isinstance(fid, str):
            combined[fid] = obj
    save_json(list(combined.values()), out_path)
    return len(combined)


async def runner(ids: List[str], out_path: str, api_key: str, concurrency: int, max_per_second: Optional[int], max_requests: Optional[int]):
    existing_list, existing_ids = load_existing(out_path)
    ids_to_fetch = [x for x in ids if x not in existing_ids]
    if max_requests and max_requests > 0:
        ids_to_fetch = ids_to_fetch[:max_requests]
    if not ids_to_fetch:
        total = combine_and_save(out_path, existing_list, {})
        logging.info(f"Nada nuevo que descargar. Total en fichero: {total}")
        return

    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency)
    headers = {
        "accept": "application/json",
        "X-Places-Api-Version": "2025-06-17",
        "authorization": f"Bearer {api_key}",
        "accept-encoding": "gzip, deflate",
    }

    results: Dict[str, dict] = {}
    done = 0
    total = len(ids_to_fetch)

    async with httpx.AsyncClient(limits=limits, headers=headers, http2=True) as client:
        sem = asyncio.Semaphore(concurrency)
        limiter = RateLimiter(max_per_second)

        async def task(fsq):
            nonlocal done
            async with sem:
                obj = await fetch_one(client, fsq, limiter)
                if obj:
                    results[fsq] = obj
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"\r[3/8] Progreso {done}/{total}", end="")

        await asyncio.gather(*(task(fsq) for fsq in ids_to_fetch))

    print()
    total_saved = combine_and_save(out_path, existing_list, results)
    logging.info(f"Descargados {len(results):,} nuevos. Total en {out_path}: {total_saved:,}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Descarga POIs desde IDs Foursquare")
    parser.add_argument("--city", required=True, help="osaka / istanbul / petalingjaya")
    parser.add_argument("--ids-path", help="Ruta alternativa de ids_<city>.txt; si no se indica, se usan IDs de std_clean.csv")
    parser.add_argument("--out", help="Ruta alternativa de salida")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-per-second", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=2000, help="Límite de IDs a consultar en esta ejecución (0 = sin límite)")
    args = parser.parse_args()

    cfg = get_city_config(args.city)
    out_path = args.out or f"data/raw/raw_pois_{cfg['file']}.json"

    api_key = os.getenv("FOURSQUARE_API_KEY")
    if not api_key:
        raise SystemExit("FOURSQUARE_API_KEY no encontrada en .env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.ids_path:
        ids = read_ids(args.ids_path)
    else:
        if not os.path.exists(STD_PATH):
            raise SystemExit(f"No se encontró {STD_PATH}; especifica --ids-path")
        df = pd.read_csv(STD_PATH, usecols=["venue_id", "venue_city"])
        ids = (
            df[df["venue_city"] == cfg["qid"]]["venue_id"]
            .dropna()
            .astype(str)
            .map(clean_fsq_id)
            .dropna()
            .unique()
            .tolist()
        )
        logging.info(f"{cfg['name']}: {len(ids):,} IDs tomados de std_clean.csv")
    mps = None if args.max_per_second <= 0 else args.max_per_second
    asyncio.run(runner(ids, out_path, api_key, args.concurrency, mps, args.max_requests))


if __name__ == "__main__":
    main()
