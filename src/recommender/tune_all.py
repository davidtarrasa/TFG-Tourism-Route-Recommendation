"""Run a fast tuning suite for a single city and save a combined report.

This does NOT modify `configs/recommender.toml` automatically.
It outputs a JSON with suggested values to copy into the config.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from .config import DEFAULT_CONFIG_PATH, load_config


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Tune all components (fast suite)")
    p.add_argument("--city-qid", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--embeddings-path", default=None)
    p.add_argument("--als-path", default=None)
    p.add_argument("--out", default=os.path.join("data", "reports", "tune_all.json"))
    args = p.parse_args()

    cfg = load_config(args.config, city_qid=args.city_qid)
    tune = cfg.get("tune", {})
    tune_data = tune.get("data", {})

    visits_limit = int(tune_data.get("visits_limit", 120000))
    k = int(tune_data.get("k", 20))
    test_size = int(tune_data.get("test_size", 1))
    min_train = int(tune_data.get("min_train", 2))
    max_cases = int(tune_data.get("max_cases", 150))
    seed = int(tune.get("seed", 42))

    # Output paths
    base = os.path.join("data", "reports")
    out_h = os.path.join(base, f"tune_hybrid_{args.city_qid}.json")
    out_m = os.path.join(base, f"tune_markov_{args.city_qid}.json")
    out_e = os.path.join(base, f"tune_embeddings_scoring_{args.city_qid}.json")
    out_a = os.path.join(base, f"tune_als_{args.city_qid}.json")
    out_rp = os.path.join(base, f"tune_route_planner_{args.city_qid}.json")

    # Run modules as subprocess-like imports (simple and stable on Windows).
    # We keep calls explicit so the user can also run each tuner independently.
    import subprocess
    import sys

    py = sys.executable

    def run(cmd):
        print("RUN:", " ".join(cmd))
        subprocess.check_call(cmd)

    # Hybrid weights (can use embeddings/ALS for scoring if available)
    cmd = [
        py,
        "-m",
        "src.recommender.tune_hybrid",
        "--city-qid",
        args.city_qid,
        "--visits-limit",
        str(visits_limit),
        "--k",
        str(k),
        "--test-size",
        str(test_size),
        "--min-train",
        str(min_train),
        "--max-cases",
        str(max_cases),
        "--seed",
        str(seed),
        "--out",
        out_h,
    ]
    if args.embeddings_path:
        cmd += ["--use-embeddings", "--embeddings-path", args.embeddings_path]
    if args.als_path:
        cmd += ["--use-als", "--als-path", args.als_path]
    run(cmd)

    # Markov
    run(
        [
            py,
            "-m",
            "src.recommender.tune_markov",
            "--city-qid",
            args.city_qid,
            "--visits-limit",
            str(visits_limit),
            "--k",
            str(k),
            "--test-size",
            str(test_size),
            "--min-train",
            str(min_train),
            "--max-cases",
            str(max_cases),
            "--seed",
            str(seed),
            "--out",
            out_m,
        ]
    )

    # Embeddings scoring (requires a model file)
    if args.embeddings_path:
        run(
            [
                py,
                "-m",
                "src.recommender.tune_embeddings_scoring",
                "--city-qid",
                args.city_qid,
                "--embeddings-path",
                args.embeddings_path,
                "--visits-limit",
                str(visits_limit),
                "--k",
                str(k),
                "--test-size",
                str(test_size),
                "--min-train",
                str(min_train),
                "--max-cases",
                str(max_cases),
                "--seed",
                str(seed),
                "--out",
                out_e,
            ]
        )

    # ALS (re-trains per trial). Keep max trials small to stay fast.
    run(
        [
            py,
            "-m",
            "src.recommender.tune_als",
            "--city-qid",
            args.city_qid,
            "--visits-limit",
            str(visits_limit),
            "--k",
            str(k),
            "--test-size",
            str(test_size),
            "--min-train",
            str(min_train),
            "--max-cases",
            str(max_cases),
            "--seed",
            str(seed),
            "--max-trials",
            "6",
            "--out",
            out_a,
        ]
    )

    # Route planner (uses route-level metrics). Needs optional artifacts for best realism.
    cmd = [
        py,
        "-m",
        "src.recommender.tune_route_planner",
        "--city-qid",
        args.city_qid,
        "--visits-limit",
        str(visits_limit),
        "--test-size",
        str(test_size),
        "--min-train",
        str(min_train),
        "--max-cases",
        str(min(max_cases, 120)),
        "--seed",
        str(seed),
        "--max-trials",
        "12",
        "--out",
        out_rp,
    ]
    if args.embeddings_path:
        cmd += ["--embeddings-path", args.embeddings_path]
    if args.als_path:
        cmd += ["--als-path", args.als_path]
    run(cmd)

    # Combine into a single report with "suggested config" keys.
    rep_h = _read_json(out_h)
    rep_m = _read_json(out_m)
    rep_e = _read_json(out_e) if os.path.exists(out_e) else None
    rep_a = _read_json(out_a)
    rep_rp = _read_json(out_rp)

    suggested: Dict[str, Any] = {}
    if rep_h.get("best_weights"):
        suggested["hybrid.trail_current"] = rep_h["best_weights"]
    if rep_m.get("best"):
        suggested["markov.backoff"] = rep_m["best"]["backoff"]
        suggested["markov.hub_alpha"] = rep_m["best"]["hub_alpha"]
    if rep_e and rep_e.get("best"):
        suggested["embeddings.context_n"] = rep_e["best"]["context_n"]
        suggested["embeddings.topn_score"] = rep_e["best"]["topn_score"]
        suggested["embeddings.hub_alpha"] = rep_e["best"]["hub_alpha"]
    if rep_a.get("best"):
        suggested["als.factors"] = rep_a["best"]["factors"]
        suggested["als.regularization"] = rep_a["best"]["regularization"]
        suggested["als.iterations"] = rep_a["best"]["iterations"]
        suggested["als.alpha"] = rep_a["best"]["alpha"]
    if rep_rp.get("best"):
        b = rep_rp["best"]
        suggested["route_planner.candidate_pool"] = b["candidate_pool"]
        suggested["route_planner.pair_min_km"] = b["pair_min_km"]
        suggested["route_planner.distance_weight"] = b["distance_weight"]
        suggested["route_planner.diversity_bonus"] = b["diversity_bonus"]
        suggested["route_planner.max_per_category"] = b["max_per_category"]

    out = {
        "city_qid": args.city_qid,
        "inputs": {
            "embeddings_path": args.embeddings_path,
            "als_path": args.als_path,
            "visits_limit": visits_limit,
            "k": k,
            "test_size": test_size,
            "min_train": min_train,
            "max_cases": max_cases,
            "seed": seed,
        },
        "reports": {
            "hybrid": out_h,
            "markov": out_m,
            "embeddings_scoring": out_e if os.path.exists(out_e) else None,
            "als": out_a,
            "route_planner": out_rp,
        },
        "suggested_config": suggested,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved combined report:", args.out)
    print("Suggested config updates:")
    for k, v in suggested.items():
        print("-", k, "=", v)


if __name__ == "__main__":
    main()
