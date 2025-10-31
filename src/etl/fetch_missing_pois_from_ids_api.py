# src/etl/fetch_pois_from_ids_api.py
# python src/etl/fetch_pois_from_ids_api.py --city Osaka
import os, json, argparse, asyncio, time, logging
from typing import List, Optional, Tuple, Set, Dict
from collections import deque

import httpx
from dotenv import load_dotenv

CITY_CFG = {
    "osaka": {
        "ids": "data/errors/MISSING_POIS_Osaka.txt",
        "out": "data/raw/MISSING_POIS_Osaka.json",
        "name": "Osaka",
    },
    "istanbul": {
        "ids": "data/errors/MISSING_POIS_Istanbul.txt",
        "out": "data/raw/MISSING_POIS_Istanbul.json",
        "name": "Istanbul",
    },
    "petalingjaya": {
        "ids": "data/errors/MISSING_POIS_PetalingJaya.txt",
        "out": "data/raw/MISSING_POIS_PetalingJaya.json",
        "name": "Petaling Jaya",
    },
}


API_BASE = "https://places-api.foursquare.com/places/{}"


def read_ids(path: str) -> List[str]:
    if not os.path.exists(path):
        raise SystemExit(f"No existe el fichero de IDs: {path}")
    with open(path, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    # limpiar prefijo "foursquare:" si viene en el archivo
    cleaned = [(x.replace("foursquare:", "", 1) if x.startswith("foursquare:") else x) for x in ids]
    # deduplicar conservando el orden
    seen, out = set(), []
    for x in cleaned:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def print_progress(done: int, total: int, bar_len: int = 30):
    total = max(1, total)
    filled = int(bar_len * done / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] ({done}/{total})", end="", flush=True)


class RateLimiter:
    """Limitador simple ~max_per_second usando una ventana de 1s."""
    def __init__(self, max_per_second: Optional[int]):
        self.max_per_second = max_per_second
        self.q = deque()
        self.lock = asyncio.Lock()

    async def wait(self):
        if not self.max_per_second or self.max_per_second <= 0:
            return
        async with self.lock:
            now = time.perf_counter()
            # purgar marcas > 1s
            while self.q and (now - self.q[0]) > 1.0:
                self.q.popleft()
            while len(self.q) >= self.max_per_second:
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                while self.q and (now - self.q[0]) > 1.0:
                    self.q.popleft()
            self.q.append(time.perf_counter())


def load_existing(out_path: str) -> Tuple[List[dict], Set[str]]:
    """
    Carga el JSON existente (si hay), devolviendo la lista y el set de fsq_id.
    Maneja JSON vacío/corrupto de forma segura.
    """
    existing_list: List[dict] = []
    existing_ids: Set[str] = set()
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing_list = data
                for obj in data:
                    fsq = obj.get("fsq_id")
                    if isinstance(fsq, str):
                        existing_ids.add(fsq)
        except Exception as e:
            logging.warning(f"No se pudo leer/parsear {out_path}: {e}. Se continuará como si no existiera.")
    return existing_list, existing_ids


async def fetch_one(client: httpx.AsyncClient, fsq_id: str, limiter: RateLimiter, max_retries: int = 5) -> Optional[dict]:
    url = API_BASE.format(fsq_id)
    for attempt in range(max_retries):
        await limiter.wait()
        try:
            r = await client.get(url, timeout=20.0)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 503):
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (1.5 + attempt)
                await asyncio.sleep(wait)
                continue
            logging.warning(f"[{fsq_id}] HTTP {r.status_code}: {r.text[:180]}")
            return None
        except httpx.RequestError as e:
            logging.warning(f"[{fsq_id}] Error de red: {e}. Reintentando...")
            await asyncio.sleep(1.0 + attempt * 0.5)
    return None


def combine_and_save(out_path: str, existing_list: List[dict], newly_fetched: Dict[str, dict]) -> int:
    """
    Combina lo existente + lo nuevo (dict por fsq_id) y guarda en disco.
    Devuelve el total de únicos guardados.
    """
    combined: Dict[str, dict] = {}
    for obj in existing_list:
        k = obj.get("fsq_id")
        if isinstance(k, str) and k not in combined:
            combined[k] = obj
    for k, obj in newly_fetched.items():
        if isinstance(k, str):
            combined[k] = obj  # lo nuevo sobreescribe si coincide

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(list(combined.values()), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)  # write-then-rename (más seguro)
    return len(combined)


async def runner(ids: List[str], out_path: str, api_key: str, concurrency: int, max_per_second: Optional[int]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Cargar lo ya existente para saltar IDs duplicados
    existing_list, existing_ids = load_existing(out_path)
    ids_to_fetch = [x for x in ids if x not in existing_ids]

    if not ids_to_fetch:
        logging.info(f"Todos los IDs ya están en {out_path}. No hay nada que descargar.")
        # Normalizar deduplicando, por si el JSON previo tenía duplicados
        combined_total = combine_and_save(out_path, existing_list, newly_fetched={})
        logging.info(f"JSON normalizado. Total únicos: {combined_total:,}")
        return

    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency)
    headers = {
        "accept": "application/json",
        "X-Places-Api-Version": "2025-06-17",
        "authorization": f"Bearer {api_key}",
        "accept-encoding": "gzip, deflate",
    }

    results: Dict[str, dict] = {}  # fsq_id -> objeto JSON
    done = 0
    total = len(ids_to_fetch)

    async with httpx.AsyncClient(limits=limits, headers=headers, http2=True) as client:
        sem = asyncio.Semaphore(concurrency)
        limiter = RateLimiter(max_per_second)

        async def task(fsq: str):
            async with sem:
                obj = await fetch_one(client, fsq, limiter)
                if obj:
                    results[fsq] = obj

        tasks = [asyncio.create_task(task(fsq)) for fsq in ids_to_fetch]

        try:
            # as_completed para ir procesando según acaben
            for coro in asyncio.as_completed(tasks):
                await coro
                done += 1
                if done % 10 == 0 or done == total:
                    print_progress(done, total)
        except KeyboardInterrupt:
            # Ctrl+C: guardamos inmediatamente lo ya descargado
            print("\n🟠 Interrupción detectada. Guardando progreso parcial...")
            combined_total = combine_and_save(out_path, existing_list, results)
            logging.info(
                f"Progreso parcial guardado en {out_path} — "
                f"{len(results):,} nuevos en esta sesión, {combined_total:,} total únicos."
            )
            return  # salimos sin propagar la interrupción
        # Si no hubo interrupción, continuar y guardar todo al final

    # Guardar al finalizar normalmente
    print_progress(total, total)
    print()
    combined_total = combine_and_save(out_path, existing_list, results)
    logging.info(
        f"RAW guardado: {out_path} — "
        f"{len(results):,} nuevos, {combined_total:,} total únicos; "
        f"IDs originales: {len(ids):,}, ya presentes: {len(existing_ids):,}, descargados ahora: {len(ids_to_fetch):,}"
    )
    print("✅ Descarga completada.")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Descarga RAW de POIs desde Foursquare usando IDs (rápido y concurrente)")
    parser.add_argument("--city", "-c", required=True, help="Osaka / Istanbul / Estambul / Petaling Jaya")
    parser.add_argument("--batch-limit", type=int, default=0, help="Limitar # de IDs (0 = todos)")
    parser.add_argument("--concurrency", type=int, default=12, help="Peticiones en paralelo")
    parser.add_argument("--max-per-second", type=int, default=0, help="Techo de peticiones por segundo (0 = sin tope)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fetch_pois_from_ids_api_raw_fast")

    key = args.city.strip().lower()
    cfg = CITY_CFG.get(key) or CITY_CFG.get(key.replace(" ", ""))
    if not cfg:
        raise SystemExit(f"Ciudad no reconocida. Opciones: {sorted(set(v['name'] for v in CITY_CFG.values()))}")

    api_key = os.getenv("FOURSQUARE_API_KEY")
    if not api_key:
        raise SystemExit("FOURSQUARE_API_KEY no encontrada en .env")

    ids = read_ids(cfg["ids"])
    if args.batch_limit and args.batch_limit > 0:
        ids = ids[:args.batch_limit]
    log.info(f"{cfg['name']}: {len(ids):,} IDs a consultar (concurrency={args.concurrency}, max_per_second={args.max_per_second or '∞'})")

    asyncio.run(runner(ids, cfg["out"], api_key, args.concurrency, args.max_per_second))


if __name__ == "__main__":
    main()
