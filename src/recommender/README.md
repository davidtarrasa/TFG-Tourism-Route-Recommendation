Product note
------------

- CLI tools are for development, debugging, and reproducible experiments.
- Final end-user interaction is expected through web/app UI, not command lines.
- Config is city-aware: if `configs/recommender_<city_qid>.toml` exists, it is auto-loaded.

What this module does
---------------------

`src/recommender/` contains the complete recommendation stack:

1. Data access and feature preparation (`utils_db.py`, `features/`)
2. Model scoring engines (`models/`)
3. Score fusion and reranking (`scorer.py`)
4. Route construction and map export (`route_planner.py`, `route_builder.py`)
5. Multi-route product contract (`multi_route_service.py`, `multi_route_cli.py`)
6. API integration layer (`api.py`)
7. Training and tuning scripts (`train_*`, `tune_*`)

Core data sources
-----------------

- `visits`: user trajectories/check-ins (trail_id, user_id, venue_id, timestamp, city_qid)
- `pois`: POI metadata (coords, rating, price, primary_category, city/city_qid)
- `poi_categories`: full category taxonomy per POI

Recommendation engines
----------------------

- `content`: TF-IDF category profile matching
- `item`: co-visitation similarity (item-item)
- `markov`: sequential transition likelihood (POI->POI, with order-2 support)
- `embed`: Word2Vec sequence neighbors
- `als`: implicit collaborative filtering
- `hybrid`: weighted combination of all available signals
- `rrf`: Reciprocal Rank Fusion — combines scores from all 5 base engines using `1/(60 + rank_i)`; no manual weight tuning needed
- `popular`: popularity baseline — ranks candidates by raw visit frequency across all users
- `random`: trivial random baseline (control)

Main scoring pipeline (`scorer.py`)
-----------------------------------

Per request:

1. Build candidate pool in selected city.
2. Compute active engine scores (depends on available signals/artifacts).
3. Normalize and combine scores using config weights.
4. Apply business reranking:
   - distance
   - price/free filters
   - user preferences (`prefs`)
   - category intent boost/strict filtering
   - diversity controls
5. Return top-K POIs with score and metadata.

Preference handling
-------------------

`prefs` can contain:

- category-like tokens (museum, sport, cafe, etc.)
- pricing/free keywords (`free`, `cheap`, `max_price:2`, etc.)

Category intents layer (`category_intents.py`):

- maps noisy raw categories into compact intent groups
- supports:
  - `soft` mode: boost preferred intents/categories
  - `strict` mode: hard filter by preferred intents/categories

Diagnostics:

- `data/reports/diagnostics/category_intent_coverage_summary.csv`
- `data/reports/diagnostics/category_intent_full_mapping.csv`

Route generation
----------------

Two stages are used:

1. Selection stage (`route_planner.py`)
   - choose coherent POI subset with leg constraints and soft diversity
2. Ordering/render stage (`route_builder.py`)
   - NN + 2-opt ordering
   - GeoJSON export
   - Folium HTML map
   - optional road path per leg (Geoapify key if present, fallback OSRM, fallback straight edge)

Map rendering capabilities (current)
------------------------------------

- numbered route markers
- straight dashed edges
- road route overlay by segment
- segment color cascade
- layer toggles and legend in exported HTML
- basemap switch (satellite/light) in exported HTML

CLI entry points
----------------

Single-route CLI:

- `python -m src.recommender.cli ...`
- supports all engine modes and optional route export

Multi-route CLI:

- `python -m src.recommender.multi_route_cli ...`
- outputs route variants and optional per-variant maps

Multi-route product contract
----------------------------

Implemented in `multi_route_service.py`.

Possible variants:

- `history`
  - generated only if user has history in selected city
  - no location and no input prefs as dominant signals
- `inputs`
  - generated only if inputs are present (`prefs` and/or filters)
  - independent from user history
  - coverage helper tries to include requested categories when feasible
- `location`
  - generated only when `lat/lon` are provided
  - geo-first route logic with city radius profile
  - sequential growth with local constraints and model tie-break signal
- `full`
  - combines whichever signals are available
  - blended from active route variants when enough components exist
  - optional low-probability "soft surprise" replacement of one stop
    (quality-gated and geo-reasonable, flagged as `is_surprise=true`)

Hard rule:

- if user has no history and request has no inputs and no location -> validation error

API integration
---------------

`api.py` exposes:

- `POST /recommend` (single route mode)
- `POST /multi-recommend` (contract mode)
- `POST/GET/DELETE /saved-routes` (persistence for frontend)

The API also:

- fills missing lat/lon when possible
- builds route payload (`ordered_pois`, `geojson`, `total_km`) if requested
- returns omitted/warnings metadata for frontend UX

Training scripts
----------------

- `train_embeddings.py`: Word2Vec artifact per city
- `train_als.py`: ALS artifact per city

Typical outputs:

- `src/recommender/cache/word2vec_q35765.joblib`
- `src/recommender/cache/als_q35765.joblib`

Tuning scripts
--------------

Available tuners:

- `tune_hybrid.py`
- `tune_markov.py`
- `tune_embeddings_scoring.py`
- `tune_als.py`
- `tune_route_planner.py`
- `tune_all.py` (aggregated suggestions)

Suggested workflow:

1. baseline eval with fixed protocol/seed
2. run tuner(s)
3. apply config updates in city TOML
4. retrain artifacts if training hyperparams changed
5. rerun eval and compare under same protocol/seed

Benchmark (3 cities)
--------------------

`benchmark_3cities.py` automates:

- optional training (`--run-train`)
- ranking eval (`--run-eval`)
- route eval (`--run-routes`)
- summary export JSON + Markdown

Example:

- `python -m src.recommender.benchmark_3cities --run-eval --run-routes`

Configuration model
-------------------

Config loading is centralized in `config.py`:

- global fallback: `configs/recommender.toml`
- city override: `configs/recommender_<city_qid>.toml`

This controls:

- hybrid weights
- markov/embedding/als scoring knobs
- route planner parameters
- eval seeds/defaults
- filter policies
- surprise policy (`[surprise]`)

Key files to inspect
--------------------

- `src/recommender/scorer.py`
- `src/recommender/multi_route_service.py`
- `src/recommender/route_planner.py`
- `src/recommender/route_builder.py`
- `src/recommender/api.py`
- `src/recommender/eval/evaluate.py`
- `src/recommender/eval/evaluate_routes.py`
