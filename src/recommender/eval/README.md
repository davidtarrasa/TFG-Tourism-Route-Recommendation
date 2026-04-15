# Offline Evaluation

This folder contains two evaluators:

- `evaluate.py`: ranking quality (next-POI recommendation quality)
- `evaluate_routes.py`: route quality (spatial coherence + diversity)

Use both together:

- ranking tells you if recommended POIs are relevant
- route eval tells you if the produced itinerary is usable

## Protocols

Both evaluators support:

- `user`
  - split by user timeline (last N visits to test)
- `trail`
  - split inside each trail/session (last N visits to test)
- `last_trail_user` (current main protocol)
  - for each user:
    - if user has only 1 trail: keep all in train (user not evaluated)
    - if user has >=2 trails and last trail has enough POIs:
      - test = full last trail
      - train = previous trails
  - evaluation seed for test route = first POI of held-out trail
  - truth = remaining POIs in held-out trail

Important split controls:

- `--min-train`
- `--test-size`
- `--min-test-pois` (especially relevant for `last_trail_user`)

## 1) Ranking Evaluation (`evaluate.py`)

Example:

```bash
python -m src.recommender.eval.evaluate --city-qid Q35765 --protocol last_trail_user --fair --visits-limit 120000 --k 20 --test-size 1 --min-train 2 --min-test-pois 4 --max-users 300 --seed 42 --modes embed item markov als hybrid content --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_q35765_current.json
```

Reported metrics (per mode):

- `hit@k`
- `precision@k`
- `recall@k`
- `ndcg@k`
- `novelty`
- `diversity`

Notes:

- `--fair` retrains models on train split only (leak-free, slower).
- without `--fair`, cached artifacts can be loaded (faster).
- `--seed` controls sampled cases when limits apply.

## 2) Route Evaluation (`evaluate_routes.py`)

Example:

```bash
python -m src.recommender.eval.evaluate_routes --city-qid Q35765 --protocol last_trail_user --fair --k 8 --test-size 1 --min-train 2 --min-test-pois 4 --max-cases 200 --visits-limit 120000 --seed 42 --modes content item markov embed als hybrid --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_routes_q35765_current.json
```

Main aggregated outputs:

- `n_routes`
- `total_km`
- `avg_leg_km`
- `pct_legs_too_close`
- `pct_legs_too_far`
- `unique_cat_ratio`
- `cat_entropy`
- `cat_match_ratio`

## Interpreting results

Recommended process:

1. Keep protocol/seed fixed.
2. Compare mode vs mode under same split.
3. Compare old config vs new config under same split.
4. Inspect both ranking and route metrics before deciding changes.

Do not compare runs with different protocol/seed/limits as if they were equivalent.

## Output files

Typical outputs:

- `data/reports/eval_<city>.json`
- `data/reports/eval_routes_<city>.json`
- benchmark aggregate:
  - `data/reports/benchmarks/benchmark_3cities_summary.json`
  - `data/reports/benchmarks/benchmark_3cities_summary.md`
