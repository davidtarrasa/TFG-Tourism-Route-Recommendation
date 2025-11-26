"""05_run_bert_classifier.py: ejecuta el clasificador BERT de categorías/precio."""
import runpy
import sys
import types
import os
import importlib.machinery
from pathlib import Path

BERT_OUT = Path("data/processed/bert_category_price_labels.json")


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


def run_bert(force: bool = False):
    """Lanza el clasificador original si no existe el fichero (o si force=True)."""
    if BERT_OUT.exists() and not force:
        print(f"[5a] Labels BERT ya existen en {BERT_OUT} (usa force=True para regenerar)")
        return
    print("[5a] Ejecutando src/etl/bert_free_category_classifier_fast.py ...")
    os.environ.setdefault("ACCELERATE_DISABLE_TORCHAO", "1")
    _stub_torchao()
    try:
        runpy.run_path("src/etl/bert_free_category_classifier_fast.py", run_name="__main__")
    except Exception as e:
        print("[5a] Error ejecutando el clasificador BERT:", e)
        print("[5a] Sugerencias: 1) pip uninstall -y torchao  2) fijar versiones torch/transformers/accelerate compatibles  3) generar labels en otro entorno y copiar data/processed/bert_category_price_labels.json")
        return
    if BERT_OUT.exists():
        print(f"[5a] Generado {BERT_OUT}")
    else:
        print("[5a] Aviso: no se generó bert_category_price_labels.json. Revisa logs del clasificador.")


def main():
    run_bert(force=False)


if __name__ == "__main__":
    main()
