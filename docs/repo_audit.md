# Auditoria Tecnica del Repositorio (TFG - Tourism Route Recommendation)

Ultima actualizacion: 2026-04-18

Este documento resume el estado real del repositorio tras el desarrollo completo del TFG.

---

## 1) Estado actual — implementado y operativo end-to-end

### Datos / ETL / BD

- Pipeline ETL completo (`src/etl/01_...08_*.py`): limpieza, enriquecimiento y carga en PostgreSQL.
- Esquema SQL (`sql/schema.sql`) con tablas `visits`, `pois`, `poi_categories`.
- La tabla `saved_routes` se crea on-demand desde la API (`src/recommender/api.py`).
- Filtro por ciudad via `city_qid` (Wikidata QID) indexado.
- 3 ciudades operativas: Osaka (Q35765), Istanbul (Q406), Petaling Jaya (Q864965).
- ~200 000 check-ins reales (Foursquare Semantic Trails 2018).

### Motores de recomendacion (9 modos)

| Modo | Tipo | Artefacto |
|------|------|-----------|
| `content` | TF-IDF sobre categorias POI | no |
| `item` | co-visitacion (item-item) | no |
| `markov` | transiciones secuenciales orden 1/2 | no |
| `embed` | Word2Vec sobre trails | `.joblib` por ciudad |
| `als` | ALS implicito (factorizacion) | `.joblib` por ciudad |
| `hybrid` | fusion ponderada de los 5 anteriores | config TOML |
| `rrf` | Reciprocal Rank Fusion automatica (k=30 por defecto) | no |
| `popular` | baseline por frecuencia de visitas | no |
| `random` | control aleatorio | no |

### Scoring y ruta

- `scorer.py`: normaliza, fusiona y aplica reranking (distancia, precio, diversidad, prefs).
- `route_planner.py`: seleccion greedy con restricciones de tramos.
- `route_builder.py`: ordenacion NN + 2-opt, GeoJSON, Folium HTML.
- Overlay de calles via OSRM (frontend) y Geoapify (si clave presente).

### Evaluacion offline

- Protocolo principal: `last_trail_user` con `--fair` (reentrenamiento sobre split de train).
- Split cold/warm: < 5 visitas TRAIN = cold, >= 5 = warm.
- Metricas: Hit@K, Precision@K, Recall@K, nDCG@K + variantes por categoria + Novelty + Diversity.
- Benchmark automatizado 3 ciudades: `benchmark_3cities.py`.
- Figuras de tesis generadas con `scripts/generate_tfg_figures.py` (set principal `fig_01`-`fig_26` + variantes auxiliares).

### Multi-ruta y API

- `multi_route_service.py`: genera 4 variantes (`history`, `inputs`, `location`, `full`) en una sola peticion.
- FastAPI con endpoints `/recommend`, `/multi-recommend`, `/saved-routes`.
- Persistencia de rutas en PostgreSQL (`saved_routes`) con fallback a localStorage en frontend.

### Frontend

- SPA HTML/CSS/JS vanilla: selector de ciudad, slider de paradas, preferencias, mapa Leaflet.
- Variantes de ruta con tabs, leyenda por tramo, fullscreen, exportar JSON, guardar.

---

## 2) Configuracion

- Config global: `configs/recommender.toml`.
- Override por ciudad: `configs/recommender_<city_qid>.toml` (Osaka, Istanbul, Petaling Jaya).
- Carga centralizada en `src/recommender/config.py`.
- Contiene: pesos hibrido, hiperparametros Word2Vec/ALS, constraints de ruta, seeds de evaluacion.

Nota: `scorer.py` cachea el config en `_CFG`. En uso CLI normal (proceso nuevo por ejecucion) no hay problema. No reusar el mismo proceso tras cambiar el TOML.

---

## 3) Resultados de evaluacion (Hit@10, protocolo last_trail_user --fair)

| Motor | Osaka | Istanbul | Petaling Jaya |
|-------|-------|----------|---------------|
| rrf | **0.479** | 0.251 | **0.439** |
| markov | 0.431 | **0.294** | 0.361 |
| popular | 0.399 | 0.255 | 0.417 |
| item | 0.399 | 0.264 | 0.405 |
| hybrid | 0.406 | 0.165 | 0.358 |
| als | 0.344 | 0.118 | 0.273 |
| embed | 0.295 | 0.008 | 0.199 |
| content | 0.149 | 0.024 | 0.114 |
| random | 0.000 | 0.012 | 0.000 |

Detalle completo: `data/reports/figures/tfg/fig_12_tabla_metricas.csv`.

---

## 4) Riesgos conocidos y estado actual

### (A) Contenido "no turistico" en datos

Foursquare incluye movilidad cotidiana (transporte, plataformas, oficinas). Los filtros por
categoria (`exclude_categories` en config) excluyen los mas ruidosos. No se han implementado
penalizaciones extra por categoria; se considera aceptable para TFG con datos reales.

### (B) Leakage en evaluacion

Resuelto con `--fair`: reentrenamiento de Word2Vec y ALS sobre el split de TRAIN de cada usuario.
Content, item y Markov se construyen directamente sobre TRAIN. No hay filtracion de datos de test.

### (C) Warnings de pandas / psycopg

`pandas.read_sql` con psycopg3 puede emitir warnings sobre tipo de conector. No afecta resultados.
Se puede migrar a SQLAlchemy si se quiere eliminar el ruido en produccion.

### (D) Ordenacion de ruta y cruces

NN + 2-opt reduce cruces pero no garantiza solucion optima. Para rutas de <= 12 POIs es
suficientemente robusto. El overlay OSRM en el frontend muestra geometria real de calles.

### (E) Istanbul embeddings

En evaluacion fair con datos escasos, muchos POIs de Istanbul quedan fuera del vocabulario
Word2Vec. El OOV backoff (recae en historial del usuario) mejora Hit@10 de 0.000 a 0.008.
Se considera aceptable para TFG dado el tamano del dataset de Istanbul.

---

## 5) Tareas completadas (respecto a la auditoria inicial de 2026-02-07)

- [x] Multi-ruta por request (`history`/`inputs`/`location`/`full`) implementado y operativo.
- [x] Persistencia de rutas (`saved_routes` en PostgreSQL + localStorage frontend).
- [x] Benchmark automatizado 3 ciudades con export JSON + Markdown.
- [x] Protocolo de evaluacion justo (`--fair`, `last_trail_user`, cold/warm split).
- [x] Tuning de hiperparametros por ciudad (backoff Markov, pesos hybrid, ALS factors).
- [x] Motores adicionales: RRF (Reciprocal Rank Fusion) y Popular (baseline).
- [x] Figuras de tesis generadas (`fig_01`-`fig_26` + variantes auxiliares/mermaid).
- [x] Documentacion consolidada (READMEs por modulo, dossier completo, CLI guide).

---

## 6) Referencias de archivos clave

- Config: `configs/recommender.toml`, `configs/recommender_<qid>.toml`
- CLI: `src/recommender/cli.py`, `src/recommender/multi_route_cli.py`
- Orquestador: `src/recommender/scorer.py`
- BD: `src/recommender/utils_db.py`, `sql/schema.sql`
- Ruta: `src/recommender/route_planner.py`, `src/recommender/route_builder.py`
- Evaluacion: `src/recommender/eval/evaluate.py`, `src/recommender/eval/evaluate_routes.py`
- Benchmark: `src/recommender/benchmark_3cities.py`
- Figuras: `scripts/generate_tfg_figures.py`
- Doc CLI: `docs/recommender_cli.md`
- Dossier: `docs/tfg_dossier_completo.md`
