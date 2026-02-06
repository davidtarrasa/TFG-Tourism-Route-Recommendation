Metodología del recomendador (versión CLI)
==========================================

Objetivo
--------
Recomendar y secuenciar POIs para usuarios (registrados o nuevos) usando datos limpios en Postgres (`visits`, `pois`, `poi_categories`) y heurísticas/ML ligeras. Generar una lista top-K y, opcionalmente, un orden (ruta) que luego se puede dibujar con OSRM/Geoapify.

Entradas esperadas (CLI/API)
----------------------------
- `--user-id`: opcional. Si existe en `visits`, habilita CF/Markov personal.
- `--city`: filtra candidatos (matching con `pois.city`).
- `--city-qid`: filtra por ciudad usando el QID de Wikidata (coherente entre `visits.venue_city` y `pois.city_qid`).
- Preferencias: categorÃ­as (`--prefs "museum,food"`), precio (`price_tier`, `is_free`), nÂº de paradas (`--k`), ubicaciÃ³n actual (`--lat --lon`) o `--current-poi`.
- Modo: `--mode hybrid|markov|content|item` (para debug) y re-ranking por distancia opcional.

`--prefs` (minimalista)
-----------------------
Una sola cadena separada por comas que se mapea a filtros/boosts:
- keywords: `free|paid|cheap|mid|expensive|price:N|max_price:N`
- otros tokens: se interpretan como preferencias de categorías (se aplica un *boost* si el POI coincide con la categoría primaria o alguna categoría secundaria).

Fuentes de datos y features
---------------------------
- POIs (`pois`): lat, lon, city, rating, price_tier, is_free, primary_category + lista de categorías (`poi_categories`).
- Visitas (`visits`): trail_id, user_id, venue_id, venue_city, timestamp → secuencias por usuario/trail.
- Features a construir (features/):
  - TF-IDF de categorías por POI (usando `poi_categories.name`).
  - Co-ocurrencias POI↔POI (misma ruta/usuario) para item-item.
  - Transiciones Markov POI→POI y categoría→categoría (ordenadas por timestamp/trail).
  - Embeddings secuenciales (Word2Vec) sobre secuencias de POIs como alternativa “state of the art” ligera.
  - Opcional: matriz usuario-POI binaria para ALS/BPR implícito.

Motores de recomendación (models/)
----------------------------------
- Content-based:
  - Similaridad coseno sobre TF-IDF de categorías; perfil de usuario = media de sus POIs visitados.
  - Re-ponderar por rating, total_ratings, price_tier/is_free según preferencias.
  - Útil para cold-start y usuarios nuevos.
- Item-item (co-visitas):
  - Matriz de co-ocurrencia; score = suma de similitudes con POIs del usuario.
  - Rápido, no requiere factorizar.
- Markov / secuencial:
  - Matriz de transiciones POI→POI; dado `current_poi`, sugerir siguientes por probabilidad y romper empates con distancia y rating.
  - Matriz categoría→categoría como fallback cuando no se conoce el POI actual.
  - Variante embeddings: Word2Vec sobre secuencias para vecinos semántico-secuenciales.
- CF implícito (opcional):
  - ALS/BPR sobre matriz usuario-POI (check-ins). Útil para usuarios con historial; requiere librería `implicit`.

Orquestación y scoring (scorer.py)
----------------------------------
1. Generar candidatos filtrando por ciudad y disponibilidad de coords.
2. Calcular scores por motor (según modo o híbrido).
3. Normalizar scores (0–1) y combinar: p.ej. `score = 0.4*item_item + 0.3*content + 0.3*markov`.
4. Re-ranking:
   - Penalizar distancia al punto actual (`lat/lon` o `current_poi`) si se pasa.
   - Filtrar/penalizar por price_tier/is_free y categorías ya vistas en la sesión.
   - Opcional: diversidad de categorías en top-K.
5. Seleccionar top-K POIs.
6. Ordenar para ruta (heurística vecino más cercano sobre los K seleccionados). Si se quiere geometría, llamar a OSRM/Geoapify para la polilínea.

CLI (cli.py, futuro)
--------------------
- Uso esperado: `python -m src.recommender.cli --user-id 5 --city osaka --k 7 --mode hybrid --current-poi <id> --lat ... --lon ... --prefs "museum,cheap"`.
- Output: tabla con POIs (score, categoría, rating, price) y, opcionalmente, orden sugerido + enlace/archivo HTML si se llama a ruteo externo.

Carpetas
--------
- models/: algoritmos (content-based, co-visitas, markov, embeddings, ALS/BPR opcional).
- features/: carga de datos y construcción de features (TF-IDF, co-ocurrencias, transiciones, Word2Vec).
- cache/: artefactos ligeros (matrices, modelos entrenados).

Notas técnicas y “state of the art” en versión ligera
-----------------------------------------------------
- Secuencial: Markov POI/categoría (Dietz 2018) y Word2Vec sobre rutas (útil en movilidad/POI recs).
- CF/Co-visitas: co-ocurrencia y/o ALS implícito para check-ins binarios (baseline robusto).
- Multi-aspecto: usa rating/price_tier/is_free en re-ranking, siguiendo enfoques multi-factor de la literatura (Hosseini 2017).
- Geográfico: re-ranking por distancia al punto actual y clustering espacial si se requiere diversidad geográfica.
