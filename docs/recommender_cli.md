# Recomendador (CLI) - Guía rápida

Modos y señales
- Modos: `hybrid`, `content`, `item`, `markov`, `embed`, `als`.
- Señales: TF-IDF de categorías, co-visitas, Markov (POI/categoría), embeddings Word2Vec, ALS implícito.
- Re-ranking: filtros de precio/gratis, penalización de distancia (`--distance-weight`), diversidad opcional (`--no-diversify`).
- `--build-route` genera HTML + GeoJSON en `data/reports/routes/`.

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
python -m src.recommender.cli --city-qid Q35765 --user-id 2725 --mode hybrid --k 10 --use-als --als-path src/recommender/cache/als_osaka.joblib --use-embeddings --embeddings-path src/recommender/cache/word2vec_osaka.joblib --lat 34.69 --lon 135.5 --distance-weight 0.3 --build-route --route-output data/reports/routes/route_osaka.html --geojson-output data/reports/routes/route_osaka.geojson
```
Nota (PowerShell/Windows): no uses `\\` para continuar lÃ­neas; usa una sola lÃ­nea (como arriba) o el backtick `` ` ``.

Evaluación offline
```bash
python -m src.recommender.eval.evaluate --city-qid Q35765 --k 20 --test-size 1 --min-train 1 --max-users 50 --use-als --als-path src/recommender/cache/als_osaka.joblib --use-embeddings --embeddings-path src/recommender/cache/word2vec_osaka.joblib
```
- Métricas: HitRate, Recall, MRR, NDCG (hold-out por usuario); imprime usuarios evaluados.

Evaluación de rutas (calidad del itinerario)
```bash
python -m src.recommender.eval.evaluate_routes --city-qid Q35765 --protocol trail --k 10 --max-cases 200 --modes content item markov embed als hybrid --use-als --als-path src/recommender/cache/als_osaka.joblib --use-embeddings --embeddings-path src/recommender/cache/word2vec_osaka.joblib --output data/reports/eval_routes_osaka.json
```
- Métricas: distancia total, distancias entre paradas (demasiado cerca/lejos), diversidad de categorías y match con perfil de categorías del usuario.

Normalización de `pois.city`
- Valores heterogéneos (ej. variantes en japonés). Unifica a `Osaka`:
```bash
python -m src.recommender.tools.normalize_cities --map osaka
```
- Usa DSN por defecto (`POSTGRES_DSN` o `postgresql://tfg:tfgpass@localhost:55432/tfg_routes`).
