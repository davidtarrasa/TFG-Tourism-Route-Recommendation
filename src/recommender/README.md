Nota de producto
----------------
- La CLI se usa para desarrollo, validacion y experimentacion interna.
- En la version de consumo (web/app), el usuario final no introducira comandos: toda la interaccion sera mediante interfaz.
- Configuracion por ciudad: si existe `configs/recommender_<city_qid>.toml`, se carga automaticamente; si no, se usa `configs/recommender.toml`.

Metodologia del recomendador (version CLI)
==========================================

Objetivo
--------
Recomendar y secuenciar POIs para usuarios (registrados o nuevos) usando datos en Postgres (`visits`, `pois`, `poi_categories`).

Entradas esperadas (CLI/API)
----------------------------
- `--user-id`: opcional. Si existe en `visits`, habilita personalizacion por historial.
- `--city` o `--city-qid`: filtra candidatos por ciudad.
- Preferencias: `--prefs`, precio (`max_price_tier`), gratis (`free_only`), numero de paradas (`--k`), ubicacion (`--lat --lon`) o `--current-poi`.
- Modo: `--mode hybrid|markov|content|item|embed|als`.

`--prefs` (formato)
-------------------
Cadena separada por comas:
- keywords: `free|paid|cheap|mid|expensive|price:N|max_price:N`
- otros tokens: se interpretan como preferencias de categoria/intencion.

Fuentes de datos y features
---------------------------
- POIs (`pois`): lat, lon, city, rating, price_tier, is_free, primary_category.
- Visitas (`visits`): trail_id, user_id, venue_id, venue_city, timestamp.
- Features:
  - TF-IDF de categorias (content)
  - Co-visitas POI-POI (item)
  - Transiciones Markov POI->POI y categoria->categoria
  - Embeddings secuenciales (Word2Vec)
  - ALS implicito (usuario-POI)

Motores de recomendacion
------------------------
- `content`: similitud por categorias
- `item`: co-visitas
- `markov`: siguiente POI por transicion
- `embed`: vecinos secuenciales por embeddings
- `als`: collaborative filtering implicito
- `hybrid`: fusion ponderada de senales

Orquestacion y scoring (`scorer.py`)
------------------------------------
1. Construccion de candidatos.
2. Calculo de score por motor.
3. Normalizacion y fusion de scores.
4. Re-ranking por distancia, precio/gratis, diversidad y preferencias.
5. Seleccion top-K.
6. Orden de ruta con heuristica de planificador.

Categorias generales (intents)
------------------------------
- Capa de mapeo de categorias ruidosas a intents compactos (`category_intents.py`).
- Dos modos:
  - `soft`: boost por match de categoria/intencion.
  - `strict`: filtro duro por categoria/intencion.
- Diagnosticos:
  - `data/reports/diagnostics/category_intent_coverage_summary.csv`
  - `data/reports/diagnostics/category_intent_full_mapping.csv`

Contrato multi-ruta por request
--------------------------------
Rutas posibles:
- `history`: domina historial
- `inputs`: dominan preferencias de usuario
- `location`: domina cercania geografica
- `full`: combina todo lo disponible

Reglas:
- Si falta una senal, no debe contaminar otras rutas.
- Para usuario nuevo (sin historial), no se genera `history`.
- `full` se genera cuando hay suficientes senales para combinar.

Estado actual
-------------
- Documentado: SI
- Implementado en API/CLI: SI (prototipo operativo)
- Pendiente: pulido final de UX y reglas de producto para web.

Benchmark unico 3 ciudades
--------------------------
Comando principal:
- `python -m src.recommender.benchmark_3cities --run-eval --run-routes`

Opcional entrenamiento previo:
- `python -m src.recommender.benchmark_3cities --run-train --run-eval --run-routes`

Protocolos de evaluacion
------------------------
Protocolo recomendado:
- `last_trail_user`
  - si usuario tiene 1 trail: queda en train
  - si tiene >=2 trails y el ultimo tiene >=4 POIs: ultimo a test, resto a train

Metricas ranking principales:
- `hit@k`
- `precision@k`
- `recall@k`
- `ndcg@k`
- `novelty`
- `diversity`

Nota de inferencia
------------------
- Usuario registrado: se excluyen POIs ya visitados en recomendacion final.
- Usuario nuevo: no aplica ese filtro historico.
