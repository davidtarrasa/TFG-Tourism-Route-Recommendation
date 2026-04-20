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
python -m src.recommender.eval.evaluate --city-qid Q35765 --protocol last_trail_user --fair --visits-limit 120000 --k 20 --test-size 1 --min-train 2 --min-test-pois 4 --max-users 300 --seed 42 --modes embed item markov als hybrid content random popular rrf --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_q35765_current.json
```

Reported metrics (per mode):

- `hit@k`
- `precision@k`
- `recall@k`
- `ndcg@k`
- `cat_hit@k`
- `cat_precision@k`
- `cat_recall@k`
- `cat_ndcg@k`
- `novelty`
- `diversity`

Notes:

- `random` is a trivial random baseline; `popular` ranks by visit frequency across all users.
- `rrf` applies Reciprocal Rank Fusion (`1/(rrf_k+rank)`) over all 5 base engine rankings — no trained artifact needed. `rrf_k` defaults to 30 and is configurable per city via `[eval] rrf_k` in each `configs/recommender_<qid>.toml`.
- All modes use the same fixed seed item and split for fair comparison.
- `--fair` retrains models on train split only (leak-free, slower).
- Without `--fair`, cached artifacts can be loaded (faster, but potential data leakage).
- `--seed` controls sampled cases when limits apply.
- Users with fewer than 5 train visits are classified as **cold**, the rest as **warm**. The output JSON includes a `cold_warm_breakdown` section with per-group metrics.

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

- `data/reports/eval_<city>_latest.json` (overwritten by benchmark; used by figure generator)
- `data/reports/eval_routes_<city>_latest.json`
- `data/reports/benchmarks/eval_<city>_<timestamp>.json` (immutable snapshots)
- benchmark aggregate:
  - `data/reports/benchmarks/benchmark_3cities_summary.json`
  - `data/reports/benchmarks/benchmark_3cities_summary.md`

Note: `scripts/generate_tfg_figures.py` reads `*_latest.json` files with priority. Run the benchmark before regenerating figures to ensure figures reflect the latest results.

---

## Comparativa con la literatura

La figura `data/reports/figures/tfg/fig_21_comparativa_literatura.png`
posiciona los resultados del TFG respecto a métodos publicados.

### Métodos ML tradicionales (mismo tipo que este sistema)

| Método | NDCG@10 aprox. | Fuente |
|--------|---------------|--------|
| BPR-MF | 0.061 | Survey POI 2024 (arXiv:2410.02191) |
| Markov puro | 0.068 | Massive-STEPS 2025 (arXiv:2505.11239) |
| FPMC | 0.094 | Rendle et al. WWW 2010 |
| Item-KNN | 0.105 | Survey POI 2024; MDPI IJGI 2023 |

**Este TFG (NDCG@20, media 3 ciudades):**
Item-Item = 0.172 · Markov = 0.149 · Hybrid = 0.136 · ALS = 0.104

→ Los métodos del TFG superan todos los baselines ML tradicionales de la literatura.

### Deep Learning SOTA (referencia de próximo paso)

| Método | NDCG@10 aprox. | Fuente |
|--------|---------------|--------|
| GRU4Rec | 0.172 | Hidasi et al. ICLR 2016 |
| GETNext | 0.241 | Yang et al. KDD 2022 |
| STHGCN | 0.298 | Massive-STEPS 2025 (arXiv:2505.11239) |

El mejor método del TFG (Item-Item, 0.172) iguala GRU4Rec y queda por debajo
de GETNext y STHGCN. Añadir una capa de deep learning secuencial sería
el siguiente paso natural para cerrar esa brecha, a costa de mayor
complejidad computacional y de entrenamiento.

### Nota metodológica

La comparación es orientativa, no directa:
- **TFG:** NDCG@20, protocolo `last_trail_user --fair`, tarea de
  *trail recommendation* (predecir una secuencia de K POIs).
- **Literatura:** NDCG@10, protocolo leave-one-out, tarea de
  *next-POI prediction* (predecir el siguiente POI individual).

### Referencias completas

- Massive-STEPS (2025): arXiv:2505.11239 — https://arxiv.org/abs/2505.11239
- Survey POI 2024: arXiv:2410.02191 — https://arxiv.org/abs/2410.02191
- FPMC: Rendle et al., WWW 2010 — https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p811.pdf
- FPMC-LR / next POI: Liu et al., IJCAI 2013 — https://www.ijcai.org/Proceedings/13/Papers/384.pdf
- GRU4Rec: Hidasi et al., ICLR 2016
- GETNext: Yang et al., KDD 2022
- Survey DL POI: Lim et al., Neurocomputing 2022 — https://doi.org/10.1016/j.neucom.2021.09.014
- MDPI IJGI 2023: https://www.mdpi.com/2220-9964/12/10/431
