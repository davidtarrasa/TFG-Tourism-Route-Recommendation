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


Plan de "multi-ruta" por request (pendiente de implementar)
-----------------------------------------------------------
Idea: ante una request de recomendacion, devolver **varias rutas alternativas** (cada una con un "motivo" distinto) y, opcionalmente, una ruta "full" que combine todas las señales disponibles.

Nota importante: **no** guardamos "historial de rutas generadas". El "historial" se refiere solo a lo que ya existe en la BD (`visits`) para ese `user_id` (check-ins / secuencias historicas).

Señales posibles en una request
-------------------------------
- Historial: existe si `user_id` aparece en `visits` (hay al menos 1 visita).
- Inputs del usuario (opcionales): `prefs` / instrucciones / restricciones (p.ej. `free`, `max_price`, categorias objetivo).
- Ubicacion (opcional): `lat/lon` o `current_poi`.

Regla de oro: si una señal no viene, **no debe afectar** a las recomendaciones que no la usan (no "ensuciar" otras rutas).

Rutas a devolver (por defecto)
------------------------------
1) Ruta por historial ("history")
   - Usa solo historial (ALS / item-item / markov / content-perfil).
   - No usa ubicacion ni prefs.
   - Si el usuario no existe (sin visitas), se omite o se degrada a "popularidad" (si existe) / content generico.

2) Ruta por inputs ("inputs")
   - Usa prefs/instrucciones como señal dominante.
   - Implementacion: content-based + filtros/boosts por categorias/price/is_free + diversidad.
   - No usa historial si el usuario es nuevo. Si el usuario existe, idealmente tampoco usa historial (para aislar el efecto del input).
   - Si no hay prefs/instrucciones, se omite.

3) Ruta por ubicacion ("location")
   - Usa ubicacion como señal dominante.
   - Implementacion: ranking por distancia (y/o penalizacion fuerte) + filtros basicos (p.ej. ciudad) + diversidad.
   - Independiente del historial.
   - Si no hay `lat/lon` ni `current_poi`, se omite.

4) Ruta con todo ("full")
   - Combina historial + inputs + ubicacion.
   - Solo se genera si **estan presentes las 3** señales (historial disponible + prefs/instrucciones + ubicacion).
   - Implementacion: hibrido con pesos + reranking por distancia + filtros/boosts por prefs.
   - Si falta alguna de las 3 señales, NO se crea esta ruta (para que el nombre "full" sea consistente).

Comportamiento para usuario nuevo vs existente
----------------------------------------------
- Usuario existente:
  - Puede recibir 1..4 rutas segun señales presentes.
  - "history" siempre posible (si hay historial suficiente), "inputs" y "location" solo si vienen, "full" solo si vienen todas.

- Usuario nuevo (no existe en `visits`):
  - No se permite usar historial (no existe).
  - Debe venir al menos un input (prefs y/o ubicacion). Si no viene ninguno: devolver error claro ("missing_input") o pedir datos.
  - Puede recibir "inputs" y/o "location". "history" se omite. "full" no aplica (falta historial).

Formato de respuesta (idea)
---------------------------
Devolver algo tipo JSON (CLI puede imprimirlo o guardar a fichero) con:
- `routes[]` donde cada ruta tenga:
  - `type`: history|inputs|location|full
  - `signals_used`: {history:bool, inputs:bool, location:bool}
  - `pois`: lista ordenada de POIs con score y metadatos
  - `explanations` (breve): por que se elige cada POI (top factores)
  - `map_outputs` (opcional): html + geojson
