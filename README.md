<div align="center">

# 🗺️ Tourism Route Recommendation

**End-to-end tourism route recommender built on real Foursquare check-ins**

_3 cities · 9 recommendation engines · FastAPI · PostgreSQL · Offline evaluation_

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![Leaflet](https://img.shields.io/badge/Leaflet.js-map-199900?logo=leaflet&logoColor=white)

</div>

---

## 📋 Table of Contents

- [What This Is](#-what-this-is)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Recommendation Engines](#-recommendation-engines)
- [Training Artifacts](#-training-artifacts)
- [Evaluation](#-evaluation)
- [Benchmark Results](#-benchmark-results)
- [Full Benchmark (3 Cities)](#-full-benchmark-3-cities)
- [Figures & Visualizations](#-figures--visualizations)
- [API Reference](#-api-reference)
- [Multi-Route Contract](#-multi-route-contract)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Architecture Diagrams](#-architecture-diagrams)
- [Documentation](#-documentation)

---

## 🎯 What This Is

A complete tourism route recommendation system that takes a user's check-in history, location, and preferences and returns a personalized itinerary — rendered as an interactive map.

|                 |                                                                |
| --------------- | -------------------------------------------------------------- |
| **Dataset**     | ~1.3M Foursquare check-ins across 3 cities                     |
| **Cities**      | Osaka 🇯🇵 · Istanbul 🇹🇷 · Petaling Jaya 🇲🇾                      |
| **Engines**     | 9 recommendation modes (content → deep hybrid)                 |
| **Output**      | Interactive HTML map + GeoJSON route                           |
| **Evaluation**  | Hit@20, nDCG@20, Precision, Recall, Novelty, Diversity         |
| **Best result** | Hit@20 = **0.450** (Osaka hybrid), **0.302** Istanbul (Markov) |

---

## ⚡ Quick Start

> **Prerequisites:** Python 3.10+, Docker Desktop running

```bash
# 1. Clone & install
git clone <repo-url> && cd TFG-Tourism-Route-Recommendation
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Start database
docker compose up -d db

# 3. Load data
python src/etl/08_load_postgres.py --dsn postgresql://tfg:tfgpass@localhost:55432/tfg_routes

# 4. Start backend + frontend
.\scripts\run_api.ps1        # http://127.0.0.1:8000
.\scripts\run_frontend.ps1   # http://localhost:8081/frontend/
```

That's it. Open `http://localhost:8081/frontend/` in your browser.

---

## 📦 Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
pip install -r requirements-dev.txt   # linting, testing tools
```

```bash
# Start PostgreSQL + pgAdmin
docker compose up -d db pgadmin

# Load processed data into PostgreSQL
python src/etl/08_load_postgres.py --dsn postgresql://tfg:tfgpass@localhost:55432/tfg_routes
```

> The DSN can also be set via the environment variable `POSTGRES_DSN`.

**pgAdmin** is available at `http://localhost:5050` (credentials in `docker-compose.yml`).

---

## 🚀 Running the App

### Product mode (recommended)

```bash
# Terminal 1 — backend API
python -m uvicorn src.recommender.api:app --reload --port 8000

# Terminal 2 — static frontend
python -m http.server 8081
```

| URL                               | Description           |
| --------------------------------- | --------------------- |
| `http://localhost:8081/frontend/` | Web UI                |
| `http://127.0.0.1:8000/health`    | API health check      |
| `http://localhost:5050`           | pgAdmin (DB explorer) |

**PowerShell shortcuts:**

```powershell
.\scripts\run_api.ps1
.\scripts\run_frontend.ps1
```

### CLI mode (research / debug)

**Single route:**

```bash
python -m src.recommender.cli \
  --city-qid Q35765 --user-id 2725 --mode hybrid --k 10 \
  --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib \
  --use-als       --als-path        src/recommender/cache/als_q35765.joblib \
  --lat 34.6937 --lon 135.5023 \
  --build-route \
  --route-output  data/reports/routes/route_q35765_hybrid.html \
  --geojson-output data/reports/routes/route_q35765_hybrid.geojson
```

**Multi-route contract:**

```bash
python -m src.recommender.multi_route_cli \
  --city-qid Q35765 --user-id 2725 \
  --lat 34.6937 --lon 135.5023 --prefs "museum,park,cheap" \
  --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib \
  --use-als        --als-path        src/recommender/cache/als_q35765.joblib \
  --build-route \
  --out-dir  data/reports/routes/multi_route_q35765 \
  --out-json data/reports/multi_route_q35765.json
```

> **`--prefs` keywords:** `free | paid | cheap | mid | expensive | price:N | max_price:N` + any category intent (`food`, `culture`, `nature`, `museum`, `park`, …)

---

## 🧠 Recommendation Engines

| Engine        | Mode key  | Algorithm                                                    | Needs artifact |
| ------------- | --------- | ------------------------------------------------------------ | -------------- |
| Content-based | `content` | TF-IDF over POI categories                                   | ✗              |
| Item-Item     | `item`    | Co-visitation similarity                                     | ✗              |
| Markov        | `markov`  | Sequential transitions order-1/2 + backoff                   | ✗              |
| Embeddings    | `embed`   | Word2Vec nearest neighbors on trails                         | ✓ Word2Vec     |
| ALS           | `als`     | Implicit collaborative filtering (ALS)                       | ✓ ALS model    |
| **Hybrid**    | `hybrid`  | Weighted fusion of all 5 above (per-city tuned)              | ✓ Both         |
| RRF           | `rrf`     | Reciprocal Rank Fusion `1/(rrf_k+rank)` auto-combining all 5 | ✗              |
| Popular       | `popular` | Ranking by global visit frequency                            | ✗              |
| Random        | `random`  | Random baseline                                              | ✗              |

Re-ranking signals applied on top: **distance**, **price/free**, **category diversity**, **declared preferences**.

---

## 🔧 Training Artifacts

`content`, `item`, `markov`, `rrf` and `popular` require **no training**. The rest:

```bash
# Osaka (Q35765)
python -m src.recommender.train_embeddings \
  --city-qid Q35765 --visits-limit 200000 \
  --out src/recommender/cache/word2vec_q35765.joblib

python -m src.recommender.train_als \
  --city-qid Q35765 --visits-limit 200000 \
  --out src/recommender/cache/als_q35765.joblib
```

```bash
# Istanbul (Q406)
python -m src.recommender.train_embeddings --city-qid Q406 --visits-limit 200000 --out src/recommender/cache/word2vec_q406.joblib
python -m src.recommender.train_als        --city-qid Q406 --visits-limit 200000 --out src/recommender/cache/als_q406.joblib

# Petaling Jaya (Q864965)
python -m src.recommender.train_embeddings --city-qid Q864965 --visits-limit 200000 --out src/recommender/cache/word2vec_q864965.joblib
python -m src.recommender.train_als        --city-qid Q864965 --visits-limit 200000 --out src/recommender/cache/als_q864965.joblib
```

Artifacts are saved in `src/recommender/cache/` and loaded automatically by the API.

---

## 📊 Evaluation

### Ranking evaluation

```bash
python -m src.recommender.eval.evaluate \
  --city-qid Q35765 \
  --protocol last_trail_user --fair \
  --visits-limit 120000 --k 20 \
  --test-size 1 --min-train 2 --min-test-pois 4 --max-users 300 --seed 42 \
  --modes embed item markov als hybrid content random popular rrf \
  --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib \
  --use-als        --als-path        src/recommender/cache/als_q35765.joblib \
  --output data/reports/eval_q35765_current.json
```

### Route quality evaluation

```bash
python -m src.recommender.eval.evaluate_routes \
  --city-qid Q35765 \
  --protocol last_trail_user --fair \
  --k 8 --max-cases 200 --visits-limit 120000 \
  --test-size 1 --min-train 2 --min-test-pois 4 --seed 42 \
  --modes content item markov embed als hybrid \
  --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib \
  --use-als        --als-path        src/recommender/cache/als_q35765.joblib \
  --output data/reports/eval_routes_q35765_current.json
```

**Ranking metrics reported per mode:**

`hit@k` · `precision@k` · `recall@k` · `ndcg@k` · `cat_hit@k` · `cat_precision@k` · `cat_recall@k` · `cat_ndcg@k` · `novelty` · `diversity`

Output JSON also includes `cold_warm_breakdown` (cold = <5 train visits, warm = ≥5).

**Route quality metrics:**

`n_routes` · `total_km` · `avg_leg_km` · `pct_legs_too_close` · `pct_legs_too_far` · `unique_cat_ratio` · `cat_entropy` · `cat_match_ratio`

> **Protocol `last_trail_user --fair`:** for each user with ≥2 trails, the last trail is held out as test set. Models are retrained on train-only data (leak-free). Seed POI = first POI of the held-out trail.

---

## 🏆 Benchmark Results

**Hit@20 — `last_trail_user --fair` protocol · seed=42 · max_users=300**

| Mode       | Osaka 🇯🇵  | Istanbul 🇹🇷 | Petaling Jaya 🇲🇾 |
| ---------- | :-------: | :---------: | :--------------: |
| **hybrid** | **0.450** |    0.224    |    **0.424**     |
| rrf        |   0.443   |  **0.247**  |    **0.424**     |
| markov     |   0.380   |  **0.302**  |      0.333       |
| item       |   0.378   |    0.264    |      0.396       |
| popular    |   0.356   |    0.255    |      0.420       |
| als        |   0.325   |    0.075    |      0.248       |
| embed      |   0.253   |    0.004    |      0.148       |
| content    |   0.149   |    0.024    |      0.100       |
| random     |   0.000   |    0.012    |      0.000       |

Full metrics (precision, recall, nDCG, novelty, diversity, category metrics, cold/warm breakdown):
→ `data/reports/figures/tfg/fig_12_tabla_metricas.csv`
→ `data/reports/benchmarks/benchmark_3cities_summary.md`

**Comparison with literature (NDCG@10 approx.):**

| Method                   | NDCG@10   | Source                  |
| ------------------------ | --------- | ----------------------- |
| BPR-MF                   | 0.061     | Survey POI 2024         |
| Markov                   | 0.068     | Massive-STEPS 2025      |
| FPMC                     | 0.094     | Rendle et al. WWW 2010  |
| Item-KNN                 | 0.105     | Survey POI 2024         |
| **This TFG — Item-Item** | **0.172** | NDCG@20, 3-city avg     |
| GRU4Rec                  | 0.172     | Hidasi et al. ICLR 2016 |
| GETNext                  | 0.241     | Yang et al. KDD 2022    |

> Note: comparison is indicative — this TFG uses NDCG@20 + `last_trail_user` (trail recommendation), literature uses NDCG@10 + leave-one-out (next-POI prediction).

---

## 🔬 Full Benchmark (3 Cities)

```bash
# Evaluation only (pre-trained artifacts required)
python -m src.recommender.benchmark_3cities --run-eval --run-routes

# Training + evaluation (full pipeline, slowest)
python -m src.recommender.benchmark_3cities --run-train --run-eval --run-routes
```

Outputs (auto-updated after each run):

```
data/reports/benchmarks/benchmark_3cities_summary.json
data/reports/benchmarks/benchmark_3cities_summary.md
data/reports/eval_<qid>_latest.json       ← read by figure generator
data/reports/eval_routes_<qid>_latest.json
```

<details>
<summary>🔧 Hyperparameter tuning (expand)</summary>

```bash
# Tune individual components (Osaka example)
python -m src.recommender.tune_hybrid     --city-qid Q35765 --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --out data/reports/tune_hybrid_q35765.json
python -m src.recommender.tune_markov     --city-qid Q35765 --out data/reports/tune_markov_q35765.json
python -m src.recommender.tune_als        --city-qid Q35765 --max-trials 6 --out data/reports/tune_als_q35765.json
python -m src.recommender.tune_embeddings_scoring --city-qid Q35765 --embeddings-path src/recommender/cache/word2vec_q35765.joblib --out data/reports/tune_embedscore_q35765.json
python -m src.recommender.tune_route_planner --city-qid Q35765 --k 8 --max-trials 12 --max-cases 120 --embeddings-path src/recommender/cache/word2vec_q35765.joblib --als-path src/recommender/cache/als_q35765.joblib --out data/reports/tune_routepl_q35765.json

# Tune everything at once
python -m src.recommender.tune_all --city-qid Q35765 --embeddings-path src/recommender/cache/word2vec_q35765.joblib --als-path src/recommender/cache/als_q35765.joblib --out data/reports/tune_all_q35765.json
```

</details>

---

## 🎨 Figures & Visualizations

### Thesis figures (26 figures)

```bash
# Generate all 26 thesis figures
python scripts/generate_tfg_figures.py

# Generate a specific figure
python scripts/generate_tfg_figures.py --only fig_12

# Skip a figure
python scripts/generate_tfg_figures.py --skip fig_03
```

Output: `data/reports/figures/tfg/`

### Dataset figures (ER, ETL, bubble chart, coverage heatmap)

```bash
# Generate all 4 dataset figures
python scripts/generate_dataset_figures.py

# Generate a specific one
python scripts/generate_dataset_figures.py --only fig_er
python scripts/generate_dataset_figures.py --only fig_etl
python scripts/generate_dataset_figures.py --only fig_bubble
python scripts/generate_dataset_figures.py --only fig_heatmap
```

Output: `data/reports/figures/dataset/`

<details>
<summary>📁 Figure index (expand)</summary>

| Figure                 | Description                                  |
| ---------------------- | -------------------------------------------- |
| `fig_01`               | System architecture — 5-layer pipeline       |
| `fig_02`               | POI map by category (Osaka)                  |
| `fig_03`               | Check-in heatmap (interactive + static)      |
| `fig_04`               | Hexbin rating map (Osaka)                    |
| `fig_05`               | Spatial comparison 3 cities                  |
| `fig_06`               | Markov transition heatmap (categories)       |
| `fig_07`               | Markov directed graph (categories)           |
| `fig_08`               | Sankey trail transitions (interactive + PNG) |
| `fig_09`               | t-SNE Word2Vec embeddings (Osaka)            |
| `fig_10`               | ALS user-POI matrix                          |
| `fig_11`               | Hybrid weights by scenario                   |
| `fig_12`               | Full metrics table (visual + CSV)            |
| `fig_13`               | nDCG@20 grouped bar chart                    |
| `fig_14`               | Multi-metric radar chart                     |
| `fig_14b`              | Engine × metric heatmap                      |
| `fig_15`               | Precision/Recall/nDCG bars (Osaka)           |
| `fig_16`               | Category distribution (Osaka)                |
| `fig_17`               | Long-tail user activity + Lorenz curve       |
| `fig_18`               | Temporal heatmap (hour × weekday)            |
| `fig_19`               | Trail length histogram                       |
| `fig_20`               | Evaluation protocol infographic              |
| `fig_21`               | Literature comparison (nDCG)                 |
| `fig_22`               | Literature comparison (Hit@K)                |
| `fig_23`               | Markov arc map — top 40 transitions (Osaka)  |
| `fig_24`               | Geographic Markov graph (3 cities)           |
| `fig_25`               | Markov learned vs real trails (Osaka)        |
| `fig_26`               | Cold vs warm user breakdown                  |
| `fig_er_diagram`       | Entity-Relationship diagram                  |
| `fig_etl_flow`         | ETL pipeline flowchart                       |
| `fig_bubble_dataset`   | Dataset volume + category bubble chart       |
| `fig_heatmap_coverage` | Dataset coverage heatmap (city × metric)     |

</details>

---

## 🔌 API Reference

Defined in `src/recommender/api.py`. The browser **never** talks directly to PostgreSQL.

```
Frontend  →  FastAPI  →  Recommender + PostgreSQL
```

| Method   | Endpoint           | Description                                              |
| -------- | ------------------ | -------------------------------------------------------- |
| `GET`    | `/health`          | Health check                                             |
| `POST`   | `/recommend`       | Single route recommendation                              |
| `POST`   | `/multi-recommend` | Multi-variant route (history · inputs · location · full) |
| `POST`   | `/saved-routes`    | Save a route                                             |
| `GET`    | `/saved-routes`    | List saved routes                                        |
| `DELETE` | `/saved-routes`    | Delete saved routes                                      |

---

## 🗺️ Multi-Route Contract

Implemented in `src/recommender/multi_route_service.py`. One request returns up to 4 route variants:

| Variant    | When generated                     | Signal used                     |
| ---------- | ---------------------------------- | ------------------------------- |
| `history`  | User has check-in history in city  | Collaborative + sequential      |
| `inputs`   | Always (when preferences provided) | Content + declared prefs        |
| `location` | `lat`/`lon` provided               | Geo-first, local radius         |
| `full`     | Always                             | Blended — all available signals |

A soft **surprise POI** can be injected into `full` with low probability (configurable via `[surprise]` in `configs/recommender.toml`, flagged as `is_surprise=true`).

For **new users** (no history): at least `inputs` or `location` is required, otherwise the request returns a validation error.

---

## ⚙️ Configuration

```
configs/
  recommender.toml              ← global defaults
  recommender_q35765.toml       ← Osaka overrides
  recommender_q406.toml         ← Istanbul overrides
  recommender_q864965.toml      ← Petaling Jaya overrides
```

Per-city tuning decisions applied:

| City          | Key decisions                                                              |
| ------------- | -------------------------------------------------------------------------- |
| Osaka         | `context_n=2`, `rrf_k=30`, hybrid weights balanced across all engines      |
| Istanbul      | `als_factors=128`, `rrf_k=30`, hybrid weights shifted toward Markov + Item |
| Petaling Jaya | `context_n=3`, `rrf_k=30`, hybrid weights balanced                         |

Config resolution: `src/recommender/config.py` → `load_config(path, city_qid)` → auto-loads per-city TOML.

---

## 🗂️ Project Structure

```
├── configs/                    # Global + per-city TOML configs
├── data/
│   ├── processed/              # Cleaned CSVs and enriched POI JSON
│   ├── raw/                    # Raw Foursquare data (not versioned)
│   └── reports/
│       ├── benchmarks/         # Benchmark JSON + MD summaries
│       ├── figures/
│       │   ├── tfg/            # 26 thesis figures
│       │   └── dataset/        # ER, ETL, bubble, heatmap
│       └── routes/             # Sample HTML/GeoJSON routes
├── docs/                       # Extended documentation
├── frontend/                   # One-page web UI (HTML + Leaflet.js)
├── scripts/
│   ├── generate_tfg_figures.py     # Thesis figure generator (26 figs)
│   ├── generate_dataset_figures.py # Dataset figure generator (4 figs)
│   ├── run_api.ps1
│   └── run_frontend.ps1
├── sql/                        # PostgreSQL schema
└── src/
    ├── etl/                    # 8 ETL scripts (CSV → PostgreSQL)
    └── recommender/
        ├── api.py              # FastAPI app
        ├── scorer.py           # Engine orchestration + scoring
        ├── route_planner.py    # POI selection + constraints
        ├── route_builder.py    # NN + 2-opt ordering
        ├── multi_route_service.py
        ├── cache/              # Trained artifacts (.joblib)
        ├── eval/               # Ranking + route evaluators
        └── benchmark_3cities.py
```

### City QID reference

| Wikidata QID | City          | Country     |
| ------------ | ------------- | ----------- |
| `Q35765`     | Osaka         | 🇯🇵 Japan    |
| `Q406`       | Istanbul      | 🇹🇷 Turkey   |
| `Q864965`    | Petaling Jaya | 🇲🇾 Malaysia |

---

## 🏗️ Architecture Diagrams

<details>
<summary>📊 System pipeline (expand)</summary>

```mermaid
flowchart TD
    U(["👤 Usuario\nnavegador"]):::user
    W["Interfaz Web\nHTML · Leaflet.js"]
    API["FastAPI\nBackend"]
    DB[("PostgreSQL\nvisits · pois · categories")]
    REC{"Motor de\nrecomendación"}
    CF["Content\nTF-IDF sobre categorías"]
    II["Item-Item\nCo-visitación"]
    MK["Markov\nTransiciones secuenciales"]
    EM["Embed\nWord2Vec sobre trails"]
    AL["ALS\nFiltrado colaborativo implícito"]
    HY["Hybrid\nFusión ponderada"]
    RRF["RRF\nReciprocal Rank Fusion"]
    POP["Popular\nBaseline por popularidad"]
    RANK["Re-ranking\n+ filtrado de ya visitados"]
    MAP[["🗺️ Mapa interactivo\n+ itinerario ordenado"]]

    U --> W --> API
    API --> DB
    API --> REC
    REC --> CF & II & MK & EM & AL & HY & RRF & POP
    CF & II & MK & EM & AL & HY & RRF & POP --> RANK
    RANK --> MAP --> U

    classDef user fill:#4a90d9,color:#fff,stroke:#2c5f8a
    classDef engine fill:#f0a500,color:#fff,stroke:#c47e00
    classDef db fill:#5cb85c,color:#fff,stroke:#3d7a3d
    class U user
    class CF,II,MK,EM,AL,HY,RRF,POP engine
    class DB db
```

</details>

<details>
<summary>🔄 Request scoring sequence (expand)</summary>

```mermaid
sequenceDiagram
    actor U as Usuario
    participant FE as Frontend
    participant API as FastAPI
    participant SC as scorer.py
    participant RP as route_planner.py
    participant RB as route_builder.py
    participant DB as PostgreSQL

    U->>FE: Introduce preferencias + ciudad
    FE->>API: POST /multi-recommend
    API->>DB: Cargar candidatos POIs (ciudad)
    DB-->>API: Pool de POIs
    API->>SC: score(user_id, prefs, city)
    SC->>SC: TF-IDF + Markov + Word2Vec + ALS
    SC->>SC: Normalizar + fusion ponderada
    SC-->>API: POIs ordenados por score
    API->>RP: plan_route(scored_pois, k, constraints)
    RP-->>API: Seleccion de k POIs
    API->>RB: build_route(poi_subset)
    RB->>RB: NN + 2-opt ordering
    RB-->>API: GeoJSON + HTML Folium
    API-->>FE: {variantes: history, inputs, location, full}
    FE-->>U: Mapa con ruta + lista POIs
```

</details>

<details>
<summary>🧠 User decision flow (expand)</summary>

```mermaid
flowchart TD
    START(["Petición de ruta\nciudad · k POIs"])
    H1{¿Usuario tiene\nhistorial de visitas?}
    H2{¿Proporciona\npreferencias de categoría?}
    H3{¿Historial rico\n≥ 10 visitas?}
    H4{¿Usuario\nrecurrente en la ciudad?}

    CT["🏷️ Motor Content\nTF-IDF · preferencias declaradas"]
    POP["🌟 Motor Item-Item\nPOIs populares en la ciudad"]
    EM["🔡 Motor Embed\nWord2Vec · trails cortos"]
    AL["🤝 Motor ALS\nFiltrado colaborativo"]
    HY["⚡ Motor Hybrid\nALS + Markov + Embed"]

    EVA["Evaluación offline\nHit@k · NDCG · Diversidad · Novelty"]
    OUT[["🗺️ Ruta generada\nk POIs ordenados + mapa"]]

    START --> H1
    H1 -- No --> H2
    H2 -- Sí --> CT
    H2 -- No --> POP
    H1 -- Sí --> H3
    H3 -- No --> EM
    H3 -- Sí --> H4
    H4 -- No --> AL
    H4 -- Sí --> HY

    CT & POP & EM & AL & HY --> EVA --> OUT

    classDef decision fill:#fff3cd,stroke:#f0a500,color:#333
    classDef engine fill:#d4edda,stroke:#28a745,color:#155724
    classDef io fill:#cce5ff,stroke:#004085,color:#004085
    class H1,H2,H3,H4 decision
    class CT,POP,EM,AL,HY engine
    class START,OUT io
```

</details>

<details>
<summary>📊 Offline evaluation system (expand)</summary>

```mermaid
flowchart TD
    DATA[("PostgreSQL\nvisits · pois · categories")]

    SPLIT{"Protocolo\nlast_trail_user"}
    TRAIN["Trails 1 a N-1\n→ TRAIN"]
    TEST["Trail N (ultimo)\n→ TEST"]
    SKIP["Usuario excluido\n(solo 1 trail)"]

    FAIR["--fair: reentrenar\nWord2Vec · ALS sobre TRAIN"]

    SEED["Seed = POI1\n(primer POI del trail test)"]
    GT["Ground truth\nPOI2 · POI3 · ... · POIn"]

    ENG["9 motores\ncontent · item · markov · embed · als\nhybrid · rrf · popular · random"]

    TOPK["Top-K candidatos\npor motor"]
    CMP["Comparar top-K\nvs ground truth"]

    M1["Hit@K · Precision@K · Recall@K · nDCG@K"]
    M2["cat_Hit@K · cat_nDCG@K · Novelty · Diversity"]

    AGG["Agregacion por usuario\ncold menos de 5 visitas · warm 5 o mas"]

    OUT[["JSON resultados\n3 ciudades x 9 motores x cold/warm"]]

    DATA --> SPLIT
    SPLIT -- "2+ trails" --> TRAIN & TEST
    SPLIT -- "1 trail" --> SKIP

    TRAIN --> FAIR --> ENG
    TEST --> SEED & GT
    SEED --> ENG

    ENG --> TOPK --> CMP
    GT --> CMP

    CMP --> M1 & M2 --> AGG --> OUT

    classDef data fill:#2E86AB,color:#fff,stroke:#1a5c7a
    classDef engine fill:#3BB273,color:#fff,stroke:#2a7a50
    classDef metric fill:#7B2D8B,color:#fff,stroke:#5a1e68
    classDef result fill:#1A1A2E,color:#fff,stroke:#0a0a1e
    classDef split fill:#F18F01,color:#fff,stroke:#c47000
    class DATA data
    class ENG engine
    class M1,M2 metric
    class OUT result
    class SPLIT split
```

</details>

---

## 📚 Documentation

| Document                                                           | Contents                                                          |
| ------------------------------------------------------------------ | ----------------------------------------------------------------- |
| [`src/recommender/README.md`](src/recommender/README.md)           | Engine internals, scoring, route building                         |
| [`src/recommender/eval/README.md`](src/recommender/eval/README.md) | Evaluation methodology, protocol details, literature comparison   |
| [`frontend/README.md`](frontend/README.md)                         | Frontend architecture, UI features                                |
| [`data/README.md`](data/README.md)                                 | Data layout, PostgreSQL schema, city QID map                      |
| [`docs/recommender_cli.md`](docs/recommender_cli.md)               | CLI quick reference for all commands                              |
| [`docs/guia_figuras_tfg.md`](docs/guia_figuras_tfg.md)             | What each of the 30 figures shows and how to read it              |
| [`docs/tfg_dossier_completo.md`](docs/tfg_dossier_completo.md)     | Full project dossier (motivation, design decisions, architecture) |

---

## 🧹 Maintenance

```powershell
# Preview what would be cleaned (safe)
.\scripts\clean_reports.ps1 -WhatIfOnly

# Actually clean (keeps benchmarks, latest snapshots, maps)
.\scripts\clean_reports.ps1
```

---

<div align="center">

_TFG · Universidad · 2025–2026_

</div>
