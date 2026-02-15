# Auditoria Tecnica del Repositorio (TFG - Tourism Route Recommendation)

Fecha: 2026-02-07

Este documento resume el estado real del repositorio, que partes estan "hechas" vs "en progreso", y los principales riesgos/incoherencias con respecto al objetivo del TFG (recomendacion + generacion/visualizacion de rutas).


## 1) Estado actual (lo que YA funciona end-to-end)

### Datos / ETL / BD
- Pipeline ETL implementado (`src/etl/01_...08_*.py`) que produce datos procesados y los carga en Postgres.
- Esquema SQL en `sql/schema.sql` con tablas:
  - `visits` (check-ins / secuencias)
  - `pois` (POIs enriquecidos)
  - `poi_categories` (categorias secundarias)
- Se ha normalizado el filtro por ciudad via `city_qid` (Wikidata) para evitar ambiguedades de nombres:
  - `pois.city_qid` existe e indexado (relevante para performance y consistencia).

### Recomendador CLI
- CLI operativa (`src/recommender/cli.py`) que:
  - consulta Postgres,
  - genera candidatos por modelo (content/item/markov/embed/als),
  - combina (hybrid),
  - aplica reranking por distancia/prefs,
  - puede planificar/ordenar itinerario y exportar HTML/GeoJSON.

### Visualizacion de rutas
- Generacion de mapas HTML (Folium) y GeoJSON a partir de la lista ordenada de POIs:
  - `src/recommender/route_builder.py`
  - `src/recommender/route_planner.py`
- Mapa por defecto en modo satelite (Esri World Imagery) con selector de capas.

### Evaluacion offline
Hay dos familias de evaluacion:
- "Next-POI" (tipo recomendador secuencial): `src/recommender/eval/evaluate.py`
  - protocolo `trail` (recomendado para Markov/embeddings),
  - protocolo `user` (hold-out por usuario).
- "Calidad de ruta" (coherencia espacial + diversidad): `src/recommender/eval/evaluate_routes.py`
  - mide distancia total, distancia media entre tramos, % tramos demasiado cerca/lejos, diversidad de categorias, etc.


## 2) Configuracion (single source of truth)

- `configs/recommender.toml` es el punto central para:
  - hiperparametros de entrenamiento (embeddings/ALS),
  - pesos del hibrido,
  - constraints de planificacion/route-quality,
  - parametros de evaluacion rapida (`[eval]` / `[eval_routes]`).

Riesgo a vigilar:
- `src/recommender/scorer.py` cachea el config globalmente (variable `_CFG`).
  - Si cambias el TOML y re-ejecutas en el mismo proceso (poco comun en CLI), no se recargara.
  - En CLI normal (nuevo proceso), no hay problema.


## 3) Lo que esta en progreso / por hacer (segun planing)

### 3.1 Producto y "request contract"
Actualmente la recomendacion se hace con un unico "modo" por ejecucion de CLI.
Pendiente (solo documentado en `src/recommender/README.md`):
- multi-ruta por request: `history`, `inputs`, `location`, `full` con reglas claras sobre senales presentes/ausentes.

### 3.2 Persistencia de usuarios / perfiles
- No hay tabla de "usuarios" ni historico de "rutas generadas" (por decision).
- Si se quisiera autentificacion/registro mas adelante, faltaria:
  - esquema adicional (users, feedback, etc.)
  - capa API/backend (opcional).

### 3.3 Modelos mas avanzados (SOTA)
Los modelos actuales son baselines fuertes para TFG:
- Markov orden 2 con backoff
- Word2Vec (secuencial)
- ALS implicito
Pendiente (opcional):
- modelos secuenciales neuronales (GRU4Rec/SASRec) o modelos contextuales mas complejos,
- ajuste sistematico de pesos (tuning) y logging de experimentos.


## 4) Incoherencias / riesgos detectados (y por que importan)

### (A) Evaluacion vs objetivo "turistico"
- El dataset de `visits` contiene mucha movilidad cotidiana (transporte, plataformas, estaciones).
- Esto puede hacer que:
  - modelos secuenciales/ALS puntuen alto en next-POI,
  - pero las rutas "parezcan poco turisticas" visualmente.
Accion (opcional, cuando se quiera): filtros/penalizaciones "turisticas" por categoria o regex.

### (B) Metricas optimistas si hay leakage
- Para ser "fair", cualquier feature entrenada debe usar solo TRAIN.
- La base del codigo respeta esto (transiciones/cooc/ALS se entrenan con TRAIN en evaluacion).
Riesgo: si se reusa un artefacto entrenado con todos los datos, se podria inflar resultados.
Recomendacion: versionar outputs en `data/reports/` y guardar siempre config + seed + limites.

### (C) Rendimiento / warnings de pandas
- `pandas.read_sql` con conexiones DBAPI puede dar warnings (sugiere SQLAlchemy).
- No rompe el sistema, pero ensucia salida y puede confundir en demo.
Recomendacion: migrar a SQLAlchemy cuando toque "pulir".

### (D) Ordenacion de ruta y cruces
- Aun con NN+2-opt, pueden aparecer cruces dependiendo del set de POIs.
- Si se necesita mas robustez:
  - aumentar pool y usar 2-opt "mejor mejora" o un TSP heuristic mas fuerte,
  - o usar OSRM/Geoapify para "route geometry" real (carreteras) y minimizar cruces en grafos reales.


## 5) Recomendacion de siguientes pasos (prioridad)

1) **Tuning rapido reproducible** (una ciudad):
   - fijar protocolo de evaluacion (p.ej. `trail`, `--fair`),
   - optimizar `hybrid.trail_current` y route_planner constraints,
   - guardar resultados y config en `data/reports/`.

2) **Benchmark interno estable**:
   - un comando que ejecute `evaluate` + `evaluate_routes` para todos los modos,
   - con los mismos limites (visits_limit/max_users/max_cases/seed) y exporte JSON.

3) **Documentacion final del sistema**:
   - consolidar `docs/recommender_cli.md` y `src/recommender/README.md` con:
     - como entrenar,
     - como evaluar,
     - como generar mapas,
     - interpretacion de metricas (que significa "buena ruta").


## 6) Referencias de archivos clave
- Config: `configs/recommender.toml`
- CLI: `src/recommender/cli.py`
- Orquestador: `src/recommender/scorer.py`
- BD: `src/recommender/utils_db.py`, `sql/schema.sql`, `src/etl/08_load_postgres.py`
- Ordenacion/Mapa: `src/recommender/route_builder.py`, `src/recommender/route_planner.py`
- Evaluacion: `src/recommender/eval/evaluate.py`, `src/recommender/eval/evaluate_routes.py`, `src/recommender/eval/route_metrics.py`
- Doc CLI: `docs/recommender_cli.md`

