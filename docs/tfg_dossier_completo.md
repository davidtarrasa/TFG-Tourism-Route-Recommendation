# TFG Tourism Route Recommendation - Dossier Tecnico Completo (Estado Actual del Repositorio)

Fecha de corte del dossier: 2026-03-12
Repositorio: `TFG-Tourism-Route-Recommendation`
Objetivo de este documento: servir como fuente unica, detallada y ordenada para redactar la memoria completa del TFG con otra IA o manualmente.

---

## 1. Resumen ejecutivo del proyecto

El TFG implementa un sistema de recomendacion de rutas turisticas sobre datos de check-ins reales y metadatos de POIs (Foursquare), con pipeline ETL, base de datos PostgreSQL, motores de recomendacion hibridos, evaluacion offline y una capa de API + frontend para uso interactivo.

El sistema actualmente cubre 3 ciudades objetivo:
- Osaka (`Q35765`)
- Istanbul (`Q406`)
- Petaling Jaya (`Q864965`)

El flujo funcional completo es:
1. Ingesta y limpieza de datos crudos.
2. Normalizacion y carga en PostgreSQL.
3. Entrenamiento de modelos por ciudad (embeddings y ALS).
4. Inferencia de recomendaciones (modos individuales + hibrido).
5. Planificacion de ruta (orden de POIs + restricciones de distancia/diversidad).
6. Visualizacion en mapa y almacenamiento de rutas.
7. Evaluacion offline de ranking y calidad de ruta.

---

## 2. Problema y objetivo tecnico

### 2.1 Problema
Dado un usuario y un contexto (ciudad, posible ubicacion inicial, preferencias), generar una recomendacion de POIs y convertirla en una ruta utilizable.

### 2.2 Objetivo tecnico principal
Disenar un recomendador hibrido que combine:
- senales historicas de usuario,
- patrones de co-visita y secuencia,
- similitud semantica de POIs,
- restricciones practicas (precio, gratis, distancia, diversidad),

para devolver rutas realistas y personalizadas.

### 2.3 Objetivo de producto (estado documentado)
Soportar escenarios de recomendacion por:
- historial,
- inputs/prefs,
- ubicacion,
- combinacion full,

y exponerlo en CLI/API/web.

---

## 3. Estructura actual del repositorio

### 3.1 Bloques principales
- `src/etl/`: pipeline de limpieza, enriquecimiento, export y carga BD.
- `src/recommender/`: logica de recomendacion, modelos, evaluacion, API.
- `frontend/`: interfaz web (HTML/CSS/JS) conectada a FastAPI.
- `configs/`: configuraciones globales y por ciudad (TOML), overrides de categorias.
- `sql/`: esquema PostgreSQL.
- `data/`: datasets raw/procesados, reportes, mapas, caches de modelos.
- `docs/`: documentacion tecnica y operativa.

### 3.2 Archivos nucleares
- ETL: `src/etl/01_clean_std.py` ... `08_load_postgres.py`
- Carga BD: `src/etl/08_load_postgres.py`
- Esquema: `sql/schema.sql`
- Scoring: `src/recommender/scorer.py`
- Planificador de rutas: `src/recommender/route_planner.py`
- Render de rutas: `src/recommender/route_builder.py`
- API: `src/recommender/api.py`
- Servicio multi-ruta: `src/recommender/multi_route_service.py`
- CLI simple: `src/recommender/cli.py`
- CLI multi-ruta: `src/recommender/multi_route_cli.py`
- Evaluacion ranking: `src/recommender/eval/evaluate.py`
- Evaluacion rutas: `src/recommender/eval/evaluate_routes.py`
- Entrenamiento embeddings: `src/recommender/train_embeddings.py`
- Entrenamiento ALS: `src/recommender/train_als.py`
- Frontend principal: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`

---

## 4. Datos y modelo de datos

## 4.1 Fuentes de datos
- Dataset de visitas/check-ins (`std_2018` procesado).
- Catalogo de POIs Foursquare (incluyendo categoria, rating, coordenadas, etc).

### 4.2 Entidades principales en PostgreSQL
- `visits`: eventos de visita (usuario, trail, fsq_id, timestamp/secuencia, ciudad).
- `pois`: metadatos de POI (fsq_id, nombre, lat/lon, rating, price, city/city_qid...).
- `poi_categories`: categorias del POI (relacion POI-categoria, potencialmente multietiqueta).

### 4.3 Clave de ciudad
Se ha consolidado filtro por `city_qid` para evitar inconsistencias por nombres multilenguaje (p. ej. Osaka vs 大阪市).

---

## 5. ETL (pipeline en orden)

Pipeline implementado por scripts en `src/etl/`:
1. Limpieza de visitas (`01_clean_std.py`)
2. Transformaciones intermedias (scripts 02..07 segun etapa)
3. Carga final a PostgreSQL (`08_load_postgres.py`)

### 5.1 Carga a BD
`08_load_postgres.py` realiza:
- aplicacion de `sql/schema.sql`,
- carga de CSV de visitas a `visits`,
- upsert de `pois`,
- insercion/upsert de `poi_categories`.

### 5.2 Diagnostico ETL para defensa
Se incluyo `src/etl/report_anexo_b.py` para:
- resumen de calidad/cobertura por ciudad,
- salida CSV/MD/PNG,
- mapa simple de trail real para visualizacion preliminar.

---

## 6. Motores de recomendacion implementados

El sistema soporta varios modos (`--mode`):
- `content`: similitud por contenido/categorias.
- `item`: co-visitation item-item.
- `markov`: transiciones secuenciales (siguiente POI probable).
- `embed`: embeddings secuenciales tipo Word2Vec sobre secuencias de visita.
- `als`: collaborative filtering implicito (factorizacion matriz usuario-POI).
- `hybrid`: fusion ponderada de senales anteriores.

### 6.1 Intencion del enfoque
No depender de un unico modelo:
- content ayuda en cold-start parcial,
- markov/item capturan estructura secuencial y co-ocurrencia,
- embeddings capturan proximidad contextual en secuencias,
- ALS aporta personalizacion fuerte cuando hay historial.

---

## 7. Scoring hibrido y restricciones

`src/recommender/scorer.py` combina puntajes por pesos configurables y aplica filtros/re-ranking.

### 7.1 Componentes de score (alto nivel)
- componente por trail/current context,
- componente por historial usuario,
- componente markov/item/content/embed/als,
- componente de distancia (si hay lat/lon o anchor).

### 7.2 Preferencias y filtros
Soporte de:
- `max_price_tier`,
- `free_only`,
- `prefs` (tokens de usuario),
- `category_mode`:
  - `soft`: boost por coincidencia de categoria/intencion,
  - `strict`: filtro duro por match.

### 7.3 No repetir visitados (usuario registrado)
Se mantiene logica para excluir visitados en recomendaciones de usuario conocido (salvo necesidades concretas de protocolo/validacion).

---

## 8. Sistema de categorias generales (intents)

Se implemento una capa para reducir el ruido de categorias Foursquare a un conjunto compacto de intenciones de usuario.

### 8.1 Modulos
- `src/recommender/category_intents.py`
- `src/recommender/classify_categories.py`
- `configs/category_intent_overrides.json`

### 8.2 Estrategia de clasificacion
Pipeline de clasificacion:
1. Reglas por keyword (matching directo/contains).
2. Fallback semantico (modelo sentence-transformers, cuando procede).
3. Overrides manuales para casos ambiguos o de negocio.

### 8.3 Salidas de diagnostico
- `data/reports/diagnostics/category_intent_coverage_summary.csv`
- `data/reports/diagnostics/category_intent_full_mapping.csv`

Estas salidas permiten auditar:
- cobertura por intent,
- categorias inconclusas,
- trazabilidad de cada categoria original -> intent final.

---

## 9. Planificacion de rutas

`src/recommender/route_planner.py` ordena POIs recomendados en una secuencia usable.

### 9.1 Heuristica de ruta
- Candidate pool configurable.
- Penalizacion por distancia y saltos no deseados.
- Reglas de separacion minima/maxima entre tramos.
- Diversidad por categoria.
- Ajustes mas estrictos cuando hay ubicacion inicial explicita.

### 9.2 Construccion/render
`src/recommender/route_builder.py` genera:
- orden de POIs,
- GeoJSON,
- HTML con mapa (markers, lineas, segmentos).

Se han iterado mejoras visuales:
- numeracion de puntos por orden,
- lineas por tramos,
- estilos de mapa,
- capas/leyendas de tramos.

---

## 10. Multi-ruta (contrato de producto)

Se introdujo una capa de servicio para variantes de ruta:
- `history`
- `inputs`
- `location`
- `full`

### 10.1 Implementacion actual
- Servicio: `src/recommender/multi_route_service.py`
- CLI: `src/recommender/multi_route_cli.py`
- API: `POST /multi-recommend`

### 10.2 Semantica objetivo
- `history`: domina historial.
- `inputs`: dominan preferencias de usuario.
- `location`: domina proximidad geografica.
- `full`: combina todo lo disponible.

### 10.3 Regla de ausencia de senales
Si falta una senal, se degrada esa variante o se omite segun contexto.

---

## 11. Entrenamiento de modelos

### 11.1 Embeddings
- Script: `src/recommender/train_embeddings.py`
- Artefacto por ciudad: `src/recommender/cache/word2vec_<city_qid>.joblib`

### 11.2 ALS
- Script: `src/recommender/train_als.py`
- Artefacto por ciudad: `src/recommender/cache/als_<city_qid>.joblib`

### 11.3 Configuracion
- Base: `configs/recommender.toml`
- Overrides por ciudad:
  - `configs/recommender_q35765.toml`
  - `configs/recommender_q406.toml`
  - `configs/recommender_q864965.toml`

---

## 12. Tuning y benchmark

### 12.1 Tuning
Se anadieron scripts para explorar hiperparametros y proponer updates de config.

### 12.2 Benchmark 3 ciudades
Script para ejecutar ciclo de entrenamiento/evaluacion/reportes sobre las tres ciudades en bloque (segun flags).

Objetivo:
- reproducibilidad,
- comparacion consistente entre ciudades,
- reduccion de errores manuales por comandos largos.

---

## 13. Evaluacion offline (estado actual)

Se manejan dos tipos de evaluacion:

### 13.1 Ranking quality
`src/recommender/eval/evaluate.py`

Protocolos de split soportados (incluyendo enfoque tutor):
- `last_trail_user`: ultimo trail por usuario en test (con reglas de elegibilidad), resto train.

Metricas actuales principales:
- `hit@k`
- `precision@k`
- `recall@k`
- `ndcg@k`
- `novelty@k`
- `diversity@k`

(MRR fue despriorizada en linea con feedback docente.)

### 13.2 Route quality
`src/recommender/eval/evaluate_routes.py`

Indicadores:
- distancia total,
- distancia media entre tramos,
- porcentaje de tramos demasiado cortos/largos,
- diversidad por categoria,
- entropia de categorias,
- ratio de match con categorias esperadas.

---

## 14. Baselines y protocolo docente

Alineacion con feedback de tutor:
- protocolo principal por ultimo trail de usuario,
- exigencia de longitud minima de ruta test,
- usuario con suficiente historial para mantener train.

Pendiente/iterable:
- fortalecer comparativas con baselines explicitos (`start_only`, `geo_nn`) dentro del mismo flujo de reporte para comparacion inmediata en memoria.

---

## 15. API FastAPI (backend)

Archivo: `src/recommender/api.py`

Endpoints activos:
- `GET /health`
- `POST /recommend`
- `POST /multi-recommend`
- `POST /saved-routes`
- `GET /saved-routes`
- `DELETE /saved-routes`

### 15.1 Capacidades
- inferencia simple y multi-variante,
- construccion opcional de ruta y geojson,
- persistencia de rutas guardadas.

### 15.2 Consideraciones
Hay warnings de `pandas.read_sql` por uso de conexion DBAPI sin SQLAlchemy; no rompe funcionalidad, pero es deuda tecnica para pulido.

---

## 16. Frontend web (estado)

Carpeta: `frontend/`

### 16.1 Funcionalidad implementada
- Formulario de demo (ciudad, paradas, preferencias, presupuesto, usuario, lat/lon).
- Selector automatico de centro de ciudad al cambiar ciudad.
- Map picker modal (seleccion de lat/lon sobre mapa).
- Integracion real con backend (`/multi-recommend`).
- Render de mapa con:
  - puntos numerados,
  - tramos por color,
  - capa de ruta por calles (OSRM) + aristas rectas,
  - leyenda de tramos.
- Rutas guardadas:
  - guardar,
  - listar,
  - reset.

### 16.2 Estado de UX
- Se ha eliminado modo mock como opcion principal (uso real).
- Persisten iteraciones de UI en curso para fullscreen/visibilidad de tramos y consistencia visual entre variantes.

---

## 17. Flujo operativo recomendado (actual)

1. Levantar PostgreSQL (Docker).
2. Cargar ETL a BD.
3. Entrenar embeddings/ALS por ciudad.
4. Ejecutar evaluacion ranking + rutas.
5. Levantar API FastAPI.
6. Levantar frontend y probar casos reales.
7. Guardar rutas candidatas para inspeccion visual.

---

## 18. Comandos representativos (resumen)

## 18.1 API y frontend
```powershell
python -m uvicorn src.recommender.api:app --reload --port 8000
python -m http.server 8081 -d frontend
```

## 18.2 Salud backend
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## 18.3 Inferencia por CLI (ejemplo)
```powershell
python -m src.recommender.cli --city-qid Q35765 --user-id 2725 --mode hybrid --k 10 --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --lat 34.6937 --lon 135.5023 --build-route --route-output data/reports/routes/route_q35765_hybrid_loc.html --geojson-output data/reports/routes/route_q35765_hybrid_loc.geojson
```

## 18.4 Evaluacion (protocolo ultimo trail)
```powershell
python -m src.recommender.eval.evaluate --city-qid Q35765 --protocol last_trail_user --k 20 --min-train 2 --max-users 300 --modes embed item markov als hybrid content --use-embeddings --embeddings-path src/recommender/cache/word2vec_q35765.joblib --use-als --als-path src/recommender/cache/als_q35765.joblib --output data/reports/eval_q35765_lasttrail.json
```

---

## 19. Cronologia de evolucion (resumen en orden)

1. Pipeline ETL y carga en PostgreSQL.
2. Recomendador base (content/item/markov/hybrid).
3. Integracion embeddings + ALS.
4. Ajustes por ciudad mediante `city_qid`.
5. Evaluacion offline multi-modo y metricas de ranking.
6. Evaluacion de calidad de ruta.
7. Tuning de hiperparametros y configuracion por ciudad.
8. Contrato multi-ruta (`history/inputs/location/full`).
9. Capa API FastAPI.
10. Frontend web conectado a backend.
11. Sistema de intents de categorias (soft/strict + overrides + diagnosticos).
12. Mejoras visuales de mapas y rutas guardadas.

---

## 20. Estado actual: que esta ya y que falta

## 20.1 Ya implementado
- ETL completo + carga BD.
- Modelos principales entrenables por ciudad.
- Inferencia CLI/API para modo simple y multi-ruta.
- Evaluacion offline (ranking + route metrics) con protocolo de ultimo trail.
- Frontend funcional conectado a backend real.
- Sistema de categorias generales con modo soft/strict.
- Persistencia de rutas guardadas.

## 20.2 En curso / pendientes recomendados
- Pulido UX de mapa (fullscreen completo, toggles de tramos en todas las variantes, consistencia visual final).
- Consolidar baselines docentes en reporte automatizado.
- Hardening de API (timeouts, errores mas explicitos, trazas limpias).
- Posible migracion de lecturas SQL a SQLAlchemy para eliminar warnings.
- Cierre documental final (figuras, tablas comparativas estables, conclusiones por ciudad).

---

## 21. Riesgos y deuda tecnica

1. Dependencia de OSRM publico para geometria por calles (latencia o fallos externos).
2. Warnings DBAPI/pandas (tecnica, no bloqueante).
3. Sensibilidad de metricas a protocolo/split y volumen de usuarios elegibles.
4. Complejidad de hiperparametros por ciudad (riesgo de sobreajuste local).
5. Necesidad de fijar seeds y versionado de artefactos para comparativas reproducibles de memoria.

---

## 22. Guion sugerido para redactar memoria (capitulos)

1. Introduccion y motivacion.
2. Estado del arte (recomendacion secuencial, rutas turisticas, hibridos).
3. Datos y problema formal.
4. Arquitectura y ETL.
5. Modelado: motores individuales + hibrido.
6. Planificacion de rutas y restricciones practicas.
7. Evaluacion offline (protocolos, metricas, baselines).
8. Implementacion de sistema (BD + API + frontend).
9. Resultados por ciudad y analisis critico.
10. Limitaciones, trabajo futuro y conclusiones.

---

## 23. Anexo de terminos clave (glosario corto)

- `POI`: Point of Interest.
- `trail`: sesion/ruta historica de visitas del usuario.
- `Top-K`: lista de K recomendaciones ordenadas por score.
- `city_qid`: identificador unico de ciudad (Wikidata-like id usado internamente).
- `soft categories`: boost de preferencias sin excluir candidatos.
- `strict categories`: filtro duro por preferencias/intenciones.
- `full route`: variante multi-ruta que combina todas las senales disponibles.

---

## 24. Nota final para uso con otra IA

Si este documento se usa como entrada para redactar la memoria completa:
- usar este texto como "fuente primaria de estado tecnico",
- complementar con tablas/figuras de `data/reports/` para resultados numericos,
- citar rutas de codigo (modulos) para trazabilidad de implementacion,
- mantener consistencia en protocolo de evaluacion (ultimo trail por usuario) para no mezclar resultados.

Fin del dossier.
