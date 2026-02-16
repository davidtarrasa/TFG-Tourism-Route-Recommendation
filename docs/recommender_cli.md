# Recomendador (CLI) - Guía rápida

Modos y señales
- Modos: `hybrid`, `content`, `item`, `markov`, `embed`, `als`.
- Señales: TF-IDF de categorías, co-visitas, Markov (POI/categoría), embeddings Word2Vec, ALS implícito.
- Re-ranking: filtros de precio/gratis, penalización de distancia (`--distance-weight`), diversidad opcional (`--no-diversify`).
- `--build-route` genera HTML + GeoJSON en `data/reports/routes/`.
- Config por ciudad (automatico): si existe `configs/recommender_<city_qid>.toml`, se usa ese archivo; si no, fallback a `configs/recommender.toml`.

Preferencias rápidas (`--prefs`)
- Una sola cadena separada por comas que se mapea a filtros/boosts básicos.
- Palabras clave soportadas: `free|paid|cheap|mid|expensive|price:N|max_price:N`.
- Cualquier otro token se interpreta como preferencia de categoría (match contra `primary_category` y `poi_categories.category_name`).
- Ejemplo: `--prefs "museum,park,free,cheap"`

Entrenar artefactos (ej. Osaka, QID Q35765)
```bash
python -m src.recommender.train_embeddings --city Osaka --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/word2vec_osaka.joblib
python -m src.recommender.train_als --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/als_osaka.joblib
```

Ejemplo de inferencia + mapa
```bash
python -m src.recommender.cli --city-qid Q35765 --user-id 2725 --mode hybrid --k 10 --use-als --als-path src/recommender/cache/als_osaka.joblib --use-embeddings --embeddings-path src/recommender/cache/word2vec_osaka.joblib --lat 34.69 --lon 135.5 --distance-weight 0.3 --build-route  --route-output data/reports/routes/route_osaka.html --geojson-output data/reports/routes/route_osaka.geojson
```
Nota (PowerShell/Windows): no uses `\\` para continuar lÃ­neas; usa una sola lÃ­nea (como arriba) o el backtick `` ` ``.
- En CLI, por ahora la ruta se dibuja solo en modo `drive`.
- Si existe `GEOAPIFY_API_KEY`, el mapa dibuja el camino real por calles. Si no, usa lineas rectas como fallback.

Evaluación offline
```bash
python -m src.recommender.eval.evaluate --city-qid Q35765 --k 20 --test-size 1 --min-train 1 --max-users 50 --seed 42 --use-als --als-path src/recommender/cache/als_osaka.joblib --use-embeddings --embeddings-path src/recommender/cache/word2vec_osaka.joblib
```
- Métricas: HitRate, Recall, MRR, NDCG (hold-out por usuario); imprime usuarios evaluados.

Evaluación de rutas (calidad del itinerario)
```bash
python -m src.recommender.eval.evaluate_routes --city-qid Q35765 --protocol trail --k 10 --max-cases 200 --seed 42 --modes content item markov embed als hybrid --use-als --als-path src/recommender/cache/als_osaka.joblib --use-embeddings --embeddings-path src/recommender/cache/word2vec_osaka.joblib --output data/reports/eval_routes_osaka.json
```
- Métricas: distancia total, distancias entre paradas (demasiado cerca/lejos), diversidad de categorías y match con perfil de categorías del usuario.

Tuning (rápido, una ciudad)
```bash
# 1) Hibrido (pesos) - no reentrena modelos "grandes"
python -m src.recommender.tune_hybrid --city-qid Q35765 --use-embeddings --embeddings-path src/recommender/cache/word2vec_osaka.joblib --use-als --out data/reports/tune_hybrid_osaka.json

# 2) Markov (backoff + hub penalty) - sin reentreno
python -m src.recommender.tune_markov --city-qid Q35765 --out data/reports/tune_markov_osaka.json

# 3) Embeddings scoring (context/topn/hub penalty) - sin reentreno
python -m src.recommender.tune_embeddings_scoring --city-qid Q35765 --embeddings-path src/recommender/cache/word2vec_osaka.joblib --out data/reports/tune_embedscore_osaka.json

# 4) ALS (hiperparámetros) - *con reentreno* dentro del tune (rápido y limitado)
python -m src.recommender.tune_als --city-qid Q35765 --max-trials 6 --out data/reports/tune_als_osaka.json

# 5) Route planner (parámetros de itinerario) - sin reentreno
python -m src.recommender.tune_route_planner --city-qid Q35765 --k 8 --max-trials 12 --max-cases 120 --embeddings-path src/recommender/cache/word2vec_osaka.joblib --als-path src/recommender/cache/als_osaka.joblib --out data/reports/tune_routepl_osaka.json

# 6) Suite completa (genera varios JSON + un "suggested_config" final)
python -m src.recommender.tune_all --city-qid Q35765 --embeddings-path src/recommender/cache/word2vec_osaka.joblib --als-path src/recommender/cache/als_osaka.joblib --out data/reports/tune_all_osaka.json
```
Notas:
- Los rangos de búsqueda por defecto están en `configs/recommender.toml` bajo `[tune.*]`.
- Antes de tunear, se guardó un snapshot de config en `configs/experiments/recommender_baseline_2026-02-07.toml`.

Benchmark único (3 ciudades)
```bash
# Solo evaluar (usa modelos ya entrenados)
python -m src.recommender.benchmark_3cities

# Entrenar + evaluar todo en una ejecución
python -m src.recommender.benchmark_3cities --train
```
Salida consolidada:
- `data/reports/benchmarks/benchmark_3cities_summary.json`
- `data/reports/benchmarks/benchmark_3cities_summary.md`

Contrato multi-ruta (prototipo CLI)
```bash
# Usuario con historial + inputs + ubicacion -> puede generar history/inputs/location/full
python -m src.recommender.multi_route_cli --city-qid Q35765 --user-id 2725 --current-poi 4b3ae51bf964a520956f25e3 --lat 34.6937 --lon 135.5023 --prefs "museum,park,cheap" --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --build-route --out-dir data/reports/routes/multi_route_osaka --out-json data/reports/multi_route_osaka.json

# Usuario nuevo (sin historial) con inputs/ubicacion -> omite history/full segun contrato
python -m src.recommender.multi_route_cli --city-qid Q35765 --user-id 99999999 --current-poi 4b3ae51bf964a520956f25e3 --lat 34.6937 --lon 135.5023 --prefs "museum,park,cheap" --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --build-route --out-dir data/reports/routes/multi_route_osaka_new --out-json data/reports/multi_route_osaka_new.json
```

Normalización de `pois.city`
- Valores heterogéneos (ej. variantes en japonés). Unifica a `Osaka`:
```bash
python -m src.recommender.tools.normalize_cities --map osaka
```
- Usa DSN por defecto (`POSTGRES_DSN` o `postgresql://tfg:tfgpass@localhost:55432/tfg_routes`).

