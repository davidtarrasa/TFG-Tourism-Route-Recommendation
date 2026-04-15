# TFG - Tourism Route Recommendation

End-to-end tourism route recommender over real check-ins (`visits`) and POI metadata (`pois`, `poi_categories`) in PostgreSQL.

The stack is currently operational in 4 layers:

1. ETL and DB load (`src/etl`, `sql/schema.sql`)
2. Recommender engines + route building (`src/recommender`)
3. Offline evaluation and benchmarking (`src/recommender/eval`, `benchmark_3cities.py`)
4. Product-like app flow (FastAPI + web UI in `frontend/`)

## Current Functional Scope

- City-level operation for `Q35765` (Osaka), `Q406` (Istanbul), `Q864965` (Petaling Jaya)
- Engines: `content`, `item`, `markov`, `embed`, `als`, `hybrid`
- Multi-route contract in one request: `history`, `inputs`, `location`, `full`
- Route planning + map rendering (HTML + GeoJSON), with street-path overlay
- Saved routes API and frontend section
- Offline evaluation (ranking + route quality), including `last_trail_user` protocol
- City-specific configs and benchmark automation

## Repository Map

- `src/etl/`: ETL scripts and loaders
- `sql/`: PostgreSQL schema
- `src/recommender/`: models, scoring, route generation, API, tuning, benchmark
- `src/recommender/eval/`: ranking and route evaluation
- `configs/`: global and per-city TOML configs
- `frontend/`: one-page UI (HTML/CSS/JS)
- `data/reports/`: generated outputs (maps, metrics, benchmark summaries)
- `docs/`: operational notes and long-form project docs

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
docker compose up -d db pgadmin
python src/etl/08_load_postgres.py --dsn postgresql://tfg:tfgpass@localhost:55432/tfg_routes
```

Notes:

- Runtime dependencies are in `requirements.txt`.
- Extra quality/dev tooling is in `requirements-dev.txt`.
- DB DSN can be provided by `--dsn` or env var `POSTGRES_DSN`.

## Run Modes

### 1) Product-like mode (recommended)

Run backend:

```bash
python -m uvicorn src.recommender.api:app --reload --port 8000
```

Run frontend:

```bash
python -m http.server 8081
```

Open:

- `http://localhost:8081/frontend/`
- API health: `http://127.0.0.1:8000/health`

PowerShell shortcuts:

- `.\scripts\run_api.ps1`
- `.\scripts\run_frontend.ps1`

## Maintenance Cleanup

Reports cleanup (keeps benchmark/diagnostics/maps and `*_latest` snapshots):

```powershell
.\scripts\clean_reports.ps1 -WhatIfOnly
.\scripts\clean_reports.ps1
```

### 2) CLI mode (research/debug)

Single-route recommendation:

```bash
python -m src.recommender.cli --city-qid Q35765 --user-id 2725 --mode hybrid --k 10 --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --lat 34.6937 --lon 135.5023 --build-route --route-output data/reports/routes/route_q35765_hybrid.html --geojson-output data/reports/routes/route_q35765_hybrid.geojson
```

Multi-route contract test:

```bash
python -m src.recommender.multi_route_cli --city-qid Q35765 --user-id 2725 --lat 34.6937 --lon 135.5023 --prefs "museum,park,cheap" --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --build-route --out-dir data/reports/routes/multi_route_q35765 --out-json data/reports/multi_route_q35765.json
```

## API Endpoints

Defined in `src/recommender/api.py`:

- `GET /health`
- `POST /recommend`
- `POST /multi-recommend`
- `POST /saved-routes`
- `GET /saved-routes`
- `DELETE /saved-routes`

The browser never talks directly to PostgreSQL.
Flow is always:
`Frontend -> FastAPI -> Recommender + Postgres`.

## Multi-Route Contract (Current Behavior)

Implemented in `src/recommender/multi_route_service.py`.

- `history`: generated only when user has history in the selected city
- `inputs`: generated from request preferences/constraints, independent from history
- `location`: generated only when `lat/lon` is provided; geo-first logic with local radius policy
- `full`: blended route combining available signals (history + inputs + location), not necessarily all three

For new users without history:

- at least one request signal (`inputs` and/or `location`) is required
- otherwise request returns validation error

## Training and Artifacts

Per-city artifacts:

- Word2Vec: `src/recommender/cache/word2vec_<qid>.joblib`
- ALS: `src/recommender/cache/als_<qid>.joblib`

Example:

```bash
python -m src.recommender.train_embeddings --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/word2vec_q35765.joblib
python -m src.recommender.train_als --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/als_q35765.joblib
```

## Evaluation

Ranking:

```bash
python -m src.recommender.eval.evaluate --city-qid Q35765 --protocol last_trail_user --fair --visits-limit 120000 --k 20 --test-size 1 --min-train 2 --min-test-pois 4 --max-users 300 --modes embed item markov als hybrid content --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_q35765_current.json
```

Route quality:

```bash
python -m src.recommender.eval.evaluate_routes --city-qid Q35765 --protocol last_trail_user --fair --k 8 --max-cases 200 --visits-limit 120000 --min-test-pois 4 --modes content item markov embed als hybrid --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_routes_q35765_current.json
```

Primary ranking metrics currently reported:

- `hit@k`
- `precision@k`
- `recall@k`
- `ndcg@k`
- `novelty`
- `diversity`

## Benchmark Automation (3 Cities)

Run ranking + route evaluation:

```bash
python -m src.recommender.benchmark_3cities --run-eval --run-routes
```

Run training + evaluation:

```bash
python -m src.recommender.benchmark_3cities --run-train --run-eval --run-routes
```

Outputs:

- `data/reports/benchmarks/benchmark_3cities_summary.json`
- `data/reports/benchmarks/benchmark_3cities_summary.md`

## Configuration and Reproducibility

- Global config: `configs/recommender.toml`
- Per-city override (auto-loaded): `configs/recommender_<city_qid>.toml`
  - `configs/recommender_q35765.toml`
  - `configs/recommender_q406.toml`
  - `configs/recommender_q864965.toml`

Config resolution is implemented in `src/recommender/config.py`.
Seed defaults are read from config when CLI seed is not passed.

## Frontend Notes

The web UI currently includes:

- city selector and auto-fill city center lat/lon
- map picker modal to choose coordinates from map
- multi-route generation and route variant switching
- map style switcher (Satellite/Light/OSM)
- fullscreen map mode
- segment-level route toggles
- local + backend saved routes handling

Frontend details: `frontend/README.md`

## Where To Read Next

- Recommender internals: `src/recommender/README.md`
- Evaluation internals: `src/recommender/eval/README.md`
- CLI usage summary: `docs/recommender_cli.md`
- Extended project dossier: `docs/tfg_dossier_completo.md`
