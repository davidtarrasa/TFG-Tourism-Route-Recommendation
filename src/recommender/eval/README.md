# Offline Evaluation

This folder contains two evaluators:

- `evaluate.py`: next-POI ranking metrics (`hit@k`, `precision@k`, `recall@k`, `ndcg@k`, `novelty`, `diversity`)
- `evaluate_routes.py`: route-level quality metrics (distance coherence + category diversity)

## 1) Ranking Evaluation

Example:

```bash
python -m src.recommender.eval.evaluate --city-qid Q35765 --protocol trail --fair --visits-limit 120000 --k 20 --test-size 1 --min-train 2 --max-users 300 --modes embed item markov als hybrid content --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_q35765_current.json
```

Main options:

- `--protocol user|trail|last_trail_user`:
  - `user`: últimas N visitas por usuario a test
  - `trail`: últimas N visitas por trail/sesión a test
  - `last_trail_user`: última ruta completa por usuario a test (seed = primer POI de esa ruta)
- `--min-test-pois`: para `last_trail_user`, tamaño mínimo del trail de test (por defecto `4`)
- `--fair`: trains models on train split only (no leakage)
- `--modes ...`: choose which engines to evaluate
- `--seed`: reproducible case sampling

## 2) Route Evaluation

Example:

```bash
python -m src.recommender.eval.evaluate_routes --city-qid Q35765 --protocol trail --k 8 --max-cases 200 --visits-limit 120000 --modes content item markov embed als hybrid --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_routes_q35765_current.json
```

Main outputs:

- `n_routes`
- `total_km`, `avg_leg_km`
- `pct_legs_too_close`, `pct_legs_too_far`
- `unique_cat_ratio`, `cat_entropy`, `cat_match_ratio`

## Notes

- For `Q406` route-eval usually works better with `--test-size 3` (with `1` many trails can be skipped because test item is already seen).
- Results are sensitive to protocol and limits. Keep them fixed when comparing experiments.
