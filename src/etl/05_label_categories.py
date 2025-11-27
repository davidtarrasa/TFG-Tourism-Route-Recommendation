"""05_label_categories.py: etiqueta categorías free/paid + tier usando BERT (wrapper simple)."""
import argparse
import glob
import json
import importlib.util
from pathlib import Path

LABELS_OUT = Path("data/processed/category_price_labels.json")
POIS_GLOB = "data/intermediate/pois_norm_*.json"
EXISTING_BERT = Path("data/processed/bert_category_price_labels.json")
BERT_RUNNER_PATH = Path("src2_prueba/05_run_bert_classifier.py")


def _load_runner():
    if not BERT_RUNNER_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("run_bert_classifier", BERT_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec) if spec else None
    if module and spec and spec.loader:
        spec.loader.exec_module(module)
        return module
    return None


def main():
    parser = argparse.ArgumentParser(description="Genera labels de categoría (free/paid + tier)")
    parser.add_argument("--force", action="store_true", help="Sobrescribir labels incluso si existen")
    parser.add_argument("--run-bert", action="store_true", help="Forzar ejecución del clasificador BERT antes de copiar")
    args = parser.parse_args()

    runner = _load_runner()
    if ((not EXISTING_BERT.exists()) or args.run_bert) and runner and hasattr(runner, "run_bert"):
        runner.run_bert(force=args.run_bert)

    if LABELS_OUT.exists() and not args.force:
        print(f"[5/8] Labels ya existen en {LABELS_OUT}. Usa --force para regenerar.")
        return

    if EXISTING_BERT.exists():
        data = json.load(EXISTING_BERT.open())
        LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(data, LABELS_OUT.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[5/8] Copiado labels existentes -> {LABELS_OUT}")
        return

    pois_files = glob.glob(POIS_GLOB)
    print("[5/8] No se encontró bert_category_price_labels.json. Ejecuta el clasificador original con los siguientes ficheros:")
    for f in pois_files:
        print(f"  - {f}")
    print("Luego copia/renombra el resultado a data/processed/bert_category_price_labels.json o vuelve a lanzar este script con --force.")


if __name__ == "__main__":
    main()
