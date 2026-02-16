"""
Run a full benchmark for 3 target cities in one command.

What it does:
- Optional training (Word2Vec + ALS) per city
- Ranking evaluation (evaluate.py)
- Route evaluation (evaluate_routes.py)
- Consolidated summary JSON + Markdown
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CitySpec:
    qid: str
    slug: str
    rank_test_size: int
    route_test_size: int


CITIES: Tuple[CitySpec, ...] = (
    CitySpec(qid="Q35765", slug="osaka", rank_test_size=1, route_test_size=1),
    CitySpec(qid="Q406", slug="istanbul", rank_test_size=3, route_test_size=3),
    CitySpec(qid="Q864965", slug="petalingjaya", rank_test_size=1, route_test_size=1),
)


def _run(cmd: List[str], label: str) -> Dict[str, object]:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print(p.stderr, file=sys.stderr)
    return {
        "label": label,
        "cmd": cmd,
        "returncode": int(p.returncode),
        "duration_sec": round(dt, 3),
    }


def _load_json(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ranking_to_mode_table(payload: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    metrics = payload.get("metrics", [])
    if not isinstance(metrics, list):
        return out
    for row in metrics:
        if not isinstance(row, dict):
            continue
        mode = str(row.get("mode", ""))
        metric = str(row.get("metric", ""))
        value = float(row.get("value", 0.0))
        if not mode or not metric:
            continue
        out.setdefault(mode, {})[metric] = value
    return out


def _routes_to_mode_table(payload: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    agg = payload.get("aggregate", [])
    if not isinstance(agg, list):
        return out
    for row in agg:
        if not isinstance(row, dict):
            continue
        mode = str(row.get("mode", ""))
        if not mode:
            continue
        out[mode] = {}
        for k, v in row.items():
            if k == "mode":
                continue
            try:
                out[mode][k] = float(v)
            except Exception:
                continue
    return out


def _write_markdown(summary_path_md: str, summary: Dict[str, object]) -> None:
    cities = summary.get("cities", [])
    lines: List[str] = []
    lines.append("# Benchmark 3 Cities Summary")
    lines.append("")
    for city in cities if isinstance(cities, list) else []:
        if not isinstance(city, dict):
            continue
        lines.append(f"## {city.get('slug', 'city')} ({city.get('qid', '')})")
        lines.append("")

        rank = city.get("ranking_by_mode", {})
        if isinstance(rank, dict) and rank:
            lines.append("### Ranking (Hit/MRR/NDCG/Recall)")
            lines.append("")
            lines.append("| mode | hit | mrr | ndcg | recall |")
            lines.append("|---|---:|---:|---:|---:|")
            for mode, vals in rank.items():
                if not isinstance(vals, dict):
                    continue
                lines.append(
                    f"| {mode} | {vals.get('hit', 0.0):.6f} | {vals.get('mrr', 0.0):.6f} | {vals.get('ndcg', 0.0):.6f} | {vals.get('recall', 0.0):.6f} |"
                )
            lines.append("")

        routes = city.get("routes_by_mode", {})
        if isinstance(routes, dict) and routes:
            lines.append("### Route Quality")
            lines.append("")
            lines.append("| mode | n_routes | total_km | avg_leg_km | pct_legs_too_close | pct_legs_too_far | cat_match_ratio |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for mode, vals in routes.items():
                if not isinstance(vals, dict):
                    continue
                lines.append(
                    f"| {mode} | {vals.get('n_routes', 0.0):.0f} | {vals.get('total_km', 0.0):.6f} | {vals.get('avg_leg_km', 0.0):.6f} | {vals.get('pct_legs_too_close', 0.0):.6f} | {vals.get('pct_legs_too_far', 0.0):.6f} | {vals.get('cat_match_ratio', 0.0):.6f} |"
                )
            lines.append("")

    with open(summary_path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Run benchmark (train/eval/routes) for Q35765, Q406, Q864965")
    p.add_argument("--run-train", action="store_true", help="Run training stage (alias of --train)")
    p.add_argument("--run-eval", action="store_true", help="Run ranking evaluation stage")
    p.add_argument("--run-routes", action="store_true", help="Run route evaluation stage")
    p.add_argument("--train", action="store_true", help="Train embeddings + ALS before evaluation")
    p.add_argument("--protocol", choices=["trail", "last_trail_user"], default="last_trail_user", help="Evaluation protocol")
    p.add_argument("--train-visits-limit", type=int, default=200000)
    p.add_argument("--eval-visits-limit", type=int, default=120000)
    p.add_argument("--k-rank", type=int, default=20)
    p.add_argument("--k-route", type=int, default=8)
    p.add_argument("--min-train", type=int, default=2)
    p.add_argument("--max-users", type=int, default=300)
    p.add_argument("--max-cases", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reports-dir", default=os.path.join("data", "reports"))
    p.add_argument("--cache-dir", default=os.path.join("src", "recommender", "cache"))
    args = p.parse_args()

    run_train = bool(args.train or args.run_train)
    # Backward-compatible behavior: if user does not specify run-eval/run-routes, run both.
    if not args.run_eval and not args.run_routes:
        run_eval = True
        run_routes = True
    else:
        run_eval = bool(args.run_eval)
        run_routes = bool(args.run_routes)

    os.makedirs(args.reports_dir, exist_ok=True)
    bench_dir = os.path.join(args.reports_dir, "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    py = sys.executable
    run_log: List[Dict[str, object]] = []
    cities_out: List[Dict[str, object]] = []

    for city in CITIES:
        emb_path = os.path.join(args.cache_dir, f"word2vec_{city.qid.lower()}.joblib")
        als_path = os.path.join(args.cache_dir, f"als_{city.qid.lower()}.joblib")
        eval_json = os.path.join(bench_dir, f"eval_{city.slug}.json")
        eval_routes_json = os.path.join(bench_dir, f"eval_routes_{city.slug}.json")

        if run_train:
            run_log.append(
                _run(
                    [
                        py,
                        "-m",
                        "src.recommender.train_embeddings",
                        "--city-qid",
                        city.qid,
                        "--visits-limit",
                        str(args.train_visits_limit),
                        "--out",
                        emb_path,
                    ],
                    f"train embeddings {city.slug}",
                )
            )
            run_log.append(
                _run(
                    [
                        py,
                        "-m",
                        "src.recommender.train_als",
                        "--city-qid",
                        city.qid,
                        "--visits-limit",
                        str(args.train_visits_limit),
                        "--out",
                        als_path,
                    ],
                    f"train als {city.slug}",
                )
            )

        if run_eval:
            run_log.append(
                _run(
                    [
                        py,
                        "-m",
                        "src.recommender.eval.evaluate",
                        "--city-qid",
                        city.qid,
                        "--protocol",
                        args.protocol,
                        "--fair",
                        "--visits-limit",
                        str(args.eval_visits_limit),
                        "--k",
                        str(args.k_rank),
                        "--test-size",
                        str(city.rank_test_size),
                        "--min-train",
                        str(args.min_train),
                        "--max-users",
                        str(args.max_users),
                        "--seed",
                        str(args.seed),
                        "--modes",
                        "embed",
                        "item",
                        "markov",
                        "als",
                        "hybrid",
                        "content",
                        "--use-embeddings",
                        "--embeddings-path",
                        emb_path,
                        "--use-als",
                        "--als-path",
                        als_path,
                        "--output",
                        eval_json,
                    ],
                    f"evaluate ranking {city.slug}",
                )
            )

        if run_routes:
            run_log.append(
                _run(
                    [
                        py,
                        "-m",
                        "src.recommender.eval.evaluate_routes",
                        "--city-qid",
                        city.qid,
                        "--protocol",
                        args.protocol,
                        "--test-size",
                        str(city.route_test_size),
                        "--k",
                        str(args.k_route),
                        "--max-cases",
                        str(args.max_cases),
                        "--visits-limit",
                        str(args.eval_visits_limit),
                        "--seed",
                        str(args.seed),
                        "--fair",
                        "--modes",
                        "content",
                        "item",
                        "markov",
                        "embed",
                        "als",
                        "hybrid",
                        "--use-embeddings",
                        "--embeddings-path",
                        emb_path,
                        "--use-als",
                        "--als-path",
                        als_path,
                        "--output",
                        eval_routes_json,
                    ],
                    f"evaluate routes {city.slug}",
                )
            )

        rank_payload = _load_json(eval_json)
        route_payload = _load_json(eval_routes_json)
        cities_out.append(
            {
                "qid": city.qid,
                "slug": city.slug,
                "files": {
                    "embeddings": emb_path,
                    "als": als_path,
                    "ranking_json": eval_json,
                    "routes_json": eval_routes_json,
                },
                "ranking_summary": rank_payload.get("summary", {}),
                "routes_summary": route_payload.get("summary", {}),
                "ranking_by_mode": _ranking_to_mode_table(rank_payload),
                "routes_by_mode": _routes_to_mode_table(route_payload),
            }
        )

    summary = {
        "benchmark": "3_cities",
        "cities": cities_out,
        "runs": run_log,
    }

    out_json = os.path.join(bench_dir, "benchmark_3cities_summary.json")
    out_md = os.path.join(bench_dir, "benchmark_3cities_summary.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_markdown(out_md, summary)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
