"""
Classify raw category catalog into compact intents.

Example:
  python -m src.recommender.classify_categories \
    --in-csv data/reports/diagnostics/categories_global.csv \
    --out-csv data/reports/diagnostics/categories_global_intents.csv \
    --out-json data/reports/diagnostics/categories_global_intents_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import pandas as pd

from .category_intents import INCONCLUSIVE, classify_category_intent


def main() -> None:
    p = argparse.ArgumentParser(description="Classify category names into 10 intents + inconclusive")
    p.add_argument("--in-csv", default="data/reports/diagnostics/categories_global.csv")
    p.add_argument("--name-col", default="name")
    p.add_argument("--count-col", default="count")
    p.add_argument("--out-csv", default="data/reports/diagnostics/categories_global_intents.csv")
    p.add_argument("--out-json", default="data/reports/diagnostics/categories_global_intents_summary.json")
    p.add_argument("--no-semantic", action="store_true", help="Disable semantic model, keyword+overrides only")
    p.add_argument("--semantic-threshold", type=float, default=0.42)
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.name_col not in df.columns:
        raise ValueError(f"Column '{args.name_col}' not found in {args.in_csv}")

    use_semantic = not args.no_semantic

    intents = []
    confs = []
    sources = []
    for nm in df[args.name_col].fillna("").astype(str):
        intent, conf, src = classify_category_intent(
            nm,
            use_semantic=use_semantic,
            semantic_threshold=args.semantic_threshold,
        )
        intents.append(intent)
        confs.append(conf)
        sources.append(src)

    out = df.copy()
    out["intent"] = intents
    out["intent_confidence"] = confs
    out["intent_source"] = sources

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")

    weight_col = args.count_col if args.count_col in out.columns else None
    if weight_col:
        weighted = out.groupby("intent")[weight_col].sum().sort_values(ascending=False).to_dict()
    else:
        weighted = {}

    summary = {
        "n_categories": int(len(out)),
        "n_inconclusive": int((out["intent"] == INCONCLUSIVE).sum()),
        "intent_counts": dict(Counter(out["intent"].tolist())),
        "intent_weighted_counts": weighted,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_json}")
    print("Summary:", summary)


if __name__ == "__main__":
    main()

