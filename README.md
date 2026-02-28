# TFG - Tourism Route Recommendation

Recommendation and route-building system over real check-ins (`visits`) and POI metadata (`pois`, `poi_categories`) in PostgreSQL.

Current project state:

- ETL pipeline implemented (`src/etl/01..08`)
- Postgres schema + load working (`sql/schema.sql`, `src/etl/08_load_postgres.py`)
- Recommender CLI fully working (`src/recommender/cli.py`)
- Models available: `content`, `item`, `markov`, `embed`, `als`, `hybrid`
- Route planning + HTML/GeoJSON output working
- Offline evaluation for ranking and routes working
- Tuning scripts working (`tune_all` + per-component tuners)

Web/API layer basic step is available in `src/recommender/api.py` (FastAPI).

## Quick Start

```bash
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
docker compose up -d db pgadmin
python src/etl/08_load_postgres.py --dsn postgresql://tfg:tfgpass@localhost:55432/tfg_routes
```

## Core Recommender Commands

Train city-specific artifacts:

```bash
python -m src.recommender.train_embeddings --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/word2vec_q35765.joblib
python -m src.recommender.train_als --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/als_q35765.joblib
```

Run inference + build route map:

```bash
python -m src.recommender.cli --city-qid Q35765 --user-id 2725 --mode hybrid --k 10 --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --lat 34.6937 --lon 135.5023 --build-route --route-output data/reports/routes/route_q35765_hybrid.html --geojson-output data/reports/routes/route_q35765_hybrid.geojson
```

## Architecture

- `frontend/`: web interface (HTML/CSS/JS)
- `src/recommender/api.py`: FastAPI backend (business logic gateway)
- `src/recommender/*`: recommender logic and models
- PostgreSQL (Docker): data store (`visits`, `pois`, `poi_categories`)

Recommended flow for product mode:

`Frontend -> FastAPI -> PostgreSQL + Recommender`

Notes:
- Browser should not connect directly to PostgreSQL.
- Backend remains necessary even with local DB, because it centralizes access, validation, and model execution.

## API (Step 1)

Run server:

```bash
python -m uvicorn src.recommender.api:app --reload --port 8000
```

Endpoints:
- `GET /health`
- `POST /recommend`
- `POST /multi-recommend`

## Frontend

Run a static server from repo root:

```bash
python -m http.server 8081
```

Open:
- `http://localhost:8081/frontend/`

PowerShell helpers:
- `.\scripts\run_api.ps1`
- `.\scripts\run_frontend.ps1`

## Evaluation

Ranking metrics:

```bash
python -m src.recommender.eval.evaluate --city-qid Q35765 --protocol last_trail_user --fair --visits-limit 120000 --k 20 --test-size 1 --min-train 2 --min-test-pois 4 --max-users 300 --modes embed item markov als hybrid content --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_q35765_current.json
```

Current primary ranking metrics:
- `hit@k`
- `precision@k`
- `recall@k`
- `ndcg@k`
- `novelty`
- `diversity`

Route metrics:

```bash
python -m src.recommender.eval.evaluate_routes --city-qid Q35765 --protocol last_trail_user --k 8 --max-cases 200 --visits-limit 120000 --min-test-pois 4 --modes content item markov embed als hybrid --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_routes_q35765_current.json
```

## Configuration

- Global config: `configs/recommender.toml`
- Per-city config (auto-loaded): `configs/recommender_<city_qid>.toml`
  - `configs/recommender_q35765.toml`
  - `configs/recommender_q406.toml`
  - `configs/recommender_q864965.toml`

## Key Paths

- Recommender docs: `src/recommender/README.md`
- CLI guide: `docs/recommender_cli.md`
- Evaluation docs: `src/recommender/eval/README.md`
- Reports/maps: `data/reports/`
- Frontend docs: `frontend/README.md`
