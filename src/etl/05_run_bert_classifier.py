# -*- coding: utf-8 -*-
"""05_run_bert_classifier.py: ejecuta el clasificador BERT de categorías/precio."""
import json
import os
import runpy
import sys
import types
import importlib.machinery
from pathlib import Path

from utils import get_city_config, load_json_list, save_json

BERT_OUT = Path("data/processed/bert_category_price_labels.json")
BERT_INPUT_DIR = Path("data/processed/foursquare")

# Listas de fuerza originales (IDs del profesor)
FREE_GT_FORCE = {
    "12094", "12064", "12000", "11045", "11128", "11001", "12047",
    "12009", "12080", "12013", "12014", "12031", "12022", "12021", "12048", "12058",
    "12017", "12101", "12106", "12108", "12099", "16032", "16041", "16020", "16026",
    "16017", "16046", "16001", "16006", "16007", "16030", "12075", "16003", "15014",
    "10047", "12003", "19042", "19046", "19050", "19022", "19047", "12086",
}
PAID_GT_FORCE = {
    "13065","13034","13003","13020","13035","13099","13272","13026","17029","13145",
    "17043","10032","17035","13064","13006","10021","17018","17048","19014","17036"
}

FREE_SUBSTRINGS = {"park", "plaza", "church", "mosque", "library", "garden", "cemetery"}
PAID_SUBSTRINGS = {"restaurant", "shop", "store", "hotel"}

FREE_THRESHOLD = 0.45
PAID_THRESHOLD = 0.55
MARGIN_MIN = 0.04


def _stub_torchao():
    """Inyecta módulos vacíos y con __spec__ para evitar fallos torchao/triton."""
    for name in [
        "torchao",
        "torchao.quantization",
        "torchao.dtypes",
        "torchao.utils",
        "torchao.float8",
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = mod


def _prepare_bert_inputs():
    """Prepara inputs para BERT desde pois_norm (categorías de la API)."""
    BERT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for city in ["osaka", "istanbul", "petalingjaya"]:
        cfg = get_city_config(city)
        norm_src = Path(f"data/intermediate/pois_norm_{cfg['file']}.json")
        if not norm_src.exists():
            continue
        data = load_json_list(str(norm_src))
        dst = BERT_INPUT_DIR / f"ALL_POIS_{cfg['file']}_prof_filtered.json"
        save_json(data, str(dst))
        print(f"[5a] Preparado para BERT: {dst} ({len(data):,} POIs) [source={norm_src}]")


def _build_force_name_sets():
    """Construye conjuntos de nombres a forzar (free/paid) usando los IDs del profesor."""
    names_free = set()
    names_paid = set()
    for city in ["osaka", "istanbul", "petalingjaya"]:
        cfg = get_city_config(city)
        prof_src = Path(f"data/raw/ALL_POIS_{cfg['file']}.json")
        if not prof_src.exists():
            continue
        data = load_json_list(str(prof_src))
        for poi in data:
            cats = poi.get("categories") or []
            for c in cats:
                cid = str(c.get("id")) if c.get("id") is not None else None
                cname = c.get("name")
                if not cid or not cname:
                    continue
                if cid in FREE_GT_FORCE:
                    names_free.add(cname)
                if cid in PAID_GT_FORCE:
                    names_paid.add(cname)
    return names_free, names_paid


def _force_labels_by_name(labels_path: Path):
    """Post-procesa labels: fuerza free/paid por nombre (para ids de la API)."""
    if not labels_path.exists():
        return
    data = json.load(labels_path.open(encoding="utf-8"))
    names_free, names_paid = _build_force_name_sets()

    # Index por nombre existente
    name_to_ids = {}
    for cid, info in data.items():
        cname = info.get("name")
        if cname:
            name_to_ids.setdefault(cname, []).append(cid)

    # Heurística de substrings
    def _heuristic_free(name: str) -> bool:
        n = name.lower()
        return any(sub in n for sub in FREE_SUBSTRINGS)

    def _heuristic_paid(name: str) -> bool:
        n = name.lower()
        return any(sub in n for sub in PAID_SUBSTRINGS)

    # Forzar sobre existentes
    for cid, info in data.items():
        cname = (info.get("name") or "").strip()
        lname = cname.lower()
        if cname in names_free or _heuristic_free(lname):
            info["free"] = True
            info["price_tier"] = 0
            info["source"] = info.get("source") or "force_name_free"
        elif cname in names_paid or _heuristic_paid(lname):
            info["free"] = False
            if not isinstance(info.get("price_tier"), int):
                info["price_tier"] = 2
            info["source"] = info.get("source") or "force_name_paid"

    # Añadir entradas para nombres forzados ausentes
    for cname in names_free:
        if cname not in name_to_ids:
            data[f"name_free::{cname}"] = {"name": cname, "free": True, "price_tier": 0, "source": "force_name_free"}
    for cname in names_paid:
        if cname not in name_to_ids:
            data[f"name_paid::{cname}"] = {"name": cname, "free": False, "price_tier": 2, "source": "force_name_paid"}

    save_json(data, str(labels_path))
    print(f"[5a] Fuerza aplicada por nombre: free={len(names_free)}, paid={len(names_paid)}")


def _patch_thresholds():
    os.environ["FREE_THRESHOLD"] = str(FREE_THRESHOLD)
    os.environ["PAID_THRESHOLD"] = str(PAID_THRESHOLD)
    os.environ["MARGIN_MIN"] = str(MARGIN_MIN)


def run_bert(force: bool = False):
    """Lanza el clasificador original si no existe el fichero (o si force=True)."""
    if BERT_OUT.exists() and not force:
        print(f"[5a] Labels BERT ya existen en {BERT_OUT} (usa force=True para regenerar)")
        return
    print("[5a] Preparando inputs para BERT (categorías API de pois_norm)...")
    _prepare_bert_inputs()
    print("[5a] Ejecutando src/etl/bert_free_category_classifier_fast.py ...")
    os.environ.setdefault("ACCELERATE_DISABLE_TORCHAO", "1")
    _patch_thresholds()
    _stub_torchao()
    runpy.run_path("src/etl/bert_free_category_classifier_fast.py", run_name="__main__")
    if BERT_OUT.exists():
        _force_labels_by_name(BERT_OUT)
        print(f"[5a] Generado {BERT_OUT}")
    else:
        print("[5a] Aviso: no se generó bert_category_price_labels.json. Revisa logs del clasificador.")


def main():
    run_bert(force=False)


if __name__ == "__main__":
    main()
