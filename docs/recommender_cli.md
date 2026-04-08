# Recomendador (CLI) - Guia rapida

Modos y senales
- Modos: `hybrid`, `content`, `item`, `markov`, `embed`, `als`.
- Senales: categorias (TF-IDF), co-visitas, Markov secuencial, embeddings Word2Vec, ALS implicito.
- Re-ranking: precio/gratis, distancia, diversidad y preferencias.
- `--build-route` genera HTML + GeoJSON en `data/reports/routes/`.
- Config por ciudad (auto): `configs/recommender_<city_qid>.toml` o fallback a `configs/recommender.toml`.

Preferencias rapidas (`--prefs`)
- Cadena separada por comas.
- Keywords: `free|paid|cheap|mid|expensive|price:N|max_price:N`.
- El resto se interpreta como preferencias de categoria/intencion.

Entrenar artefactos (ej. Osaka, Q35765)
```bash
python -m src.recommender.train_embeddings --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/word2vec_q35765.joblib
python -m src.recommender.train_als --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/als_q35765.joblib
```

Inferencia + mapa
```bash
python -m src.recommender.cli --city-qid Q35765 --user-id 2725 --mode hybrid --k 10 --use-als --als-path src/recommender/cache/als_q35765.joblib --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --lat 34.69 --lon 135.5 --distance-weight 0.3 --build-route --route-output data/reports/routes/route_q35765_hybrid.html --geojson-output data/reports/routes/route_q35765_hybrid.geojson
```
Nota PowerShell: usa una sola linea o el backtick `` ` `` para continuacion.

Evaluacion ranking
```bash
python -m src.recommender.eval.evaluate --city-qid Q35765 --protocol last_trail_user --fair --k 20 --test-size 1 --min-train 2 --min-test-pois 4 --max-users 300 --seed 42 --modes embed item markov als hybrid content --use-als --als-path src/recommender/cache/als_q35765.joblib --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --output data/reports/eval_q35765_current.json
```
Metricas: `hit@k`, `precision@k`, `recall@k`, `ndcg@k`, `novelty`, `diversity`.

Evaluacion de rutas
```bash
python -m src.recommender.eval.evaluate_routes --city-qid Q35765 --protocol last_trail_user --fair --k 8 --max-cases 200 --min-test-pois 4 --seed 42 --modes content item markov embed als hybrid --use-als --als-path src/recommender/cache/als_q35765.joblib --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --output data/reports/eval_routes_q35765_current.json
```

Tuning rapido (1 ciudad)
```bash
python -m src.recommender.tune_hybrid --city-qid Q35765 --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --out data/reports/tune_hybrid_q35765.json
python -m src.recommender.tune_markov --city-qid Q35765 --out data/reports/tune_markov_q35765.json
python -m src.recommender.tune_embeddings_scoring --city-qid Q35765 --embeddings-path src/recommender/cache/word2vec_q35765.joblib --out data/reports/tune_embedscore_q35765.json
python -m src.recommender.tune_als --city-qid Q35765 --max-trials 6 --out data/reports/tune_als_q35765.json
python -m src.recommender.tune_route_planner --city-qid Q35765 --k 8 --max-trials 12 --max-cases 120 --embeddings-path src/recommender/cache/word2vec_q35765.joblib --als-path src/recommender/cache/als_q35765.joblib --out data/reports/tune_routepl_q35765.json
python -m src.recommender.tune_all --city-qid Q35765 --embeddings-path src/recommender/cache/word2vec_q35765.joblib --als-path src/recommender/cache/als_q35765.joblib --out data/reports/tune_all_q35765.json
```

Benchmark 3 ciudades
```bash
python -m src.recommender.benchmark_3cities --run-eval --run-routes
python -m src.recommender.benchmark_3cities --run-train --run-eval --run-routes
```
Salida consolidada:
- `data/reports/benchmarks/benchmark_3cities_summary.json`
- `data/reports/benchmarks/benchmark_3cities_summary.md`

Multi-ruta (CLI)
```bash
python -m src.recommender.multi_route_cli --city-qid Q35765 --user-id 2725 --lat 34.6937 --lon 135.5023 --prefs "museum,park,cheap" --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --build-route --out-dir data/reports/routes/multi_route_q35765 --out-json data/reports/multi_route_q35765.json
```

Normalizacion de ciudad en POIs
```bash
python -m src.recommender.tools.normalize_cities --map osaka
```
