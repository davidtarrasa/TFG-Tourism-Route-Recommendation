# TFG - Tourism Route Recommendation

Sistema para recomendar y visualizar rutas turísticas a partir de visitas reales (Semantic Trails std_2018) y metadatos de POIs (Foursquare). Incluye pipeline ETL, base de datos PostgreSQL en Docker, demos de ruteo (OSRM/Geoapify) y un esqueleto de recomendador en terminal. Extensiones (API/LLM) son opcionales.

---

## Objetivos

- ETL reproducible: limpiar y enriquecer visitas/POIs con categorías, coordenadas, rating, precio.
- Persistencia: esquema relacional en Postgres (Docker) para consultas eficientes.
- Baselines de recomendación: contenido, co-visitas, Markov, heurísticas geográficas (ruta básica).
- Visualización: rutas en mapa (OSRM/Geoapify + Folium).
- Preparar el terreno para API/frontend/LLM en fases posteriores.

---

## Requisitos

- Docker y Docker Compose.
- Python 3.11+ con venv.
- Claves opcionales: `FOURSQUARE_API_KEY`, `GEOAPIFY_API_KEY` (solo si usas esas APIs).

---

## Puesta en marcha (paso a paso)

1. Configura entorno

```bash
cp .env.example .env   # y rellena claves si las necesitas
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Arranca Postgres y pgAdmin (puerto host 55432 -> contenedor 5432)

```bash
docker compose up -d db pgadmin
```

3. Carga esquema y datos procesados (incluidos en el repo)

```bash
python src/etl/08_load_postgres.py --dsn postgresql://tfg:tfgpass@localhost:55432/tfg_routes
```

4. Verifica conteos en BD

```bash
docker compose exec -T db psql -U tfg -d tfg_routes -c "SELECT COUNT(*) FROM visits; SELECT COUNT(*) FROM pois; SELECT COUNT(*) FROM poi_categories;"
```

5. pgAdmin (opcional)  
   Abrir http://localhost:8080 con credenciales de `.env`. Registrar server: host `db`, port `5432`, db `tfg_routes`, user `tfg`, pass `tfgpass`.

Notas:

- Datos procesados incluidos: `data/processed/std_clean.csv`, `data/processed/pois_enriched_*.json`, labels de categoría/precio.
- `data/raw/` queda como placeholder para datos brutos (no se versionan).

---

## Pipeline ETL (scripts 01-08)

- `01_clean_std.py`: limpia std_2018, normaliza IDs/timestamps -> `data/processed/std_clean.csv`.
- `02_extract_ids.py`: lista fsq_id por ciudad.
- `03_fetch_pois.py`: descarga POIs (API Foursquare).
- `04_normalize_pois.py`: fusiona POIs (profesor + API) a esquema canónico.
- `05_label_categories.py`: copia/etiqueta categorías (free/paid + price tier).
- `06_impute_pois.py`: imputación de price/rating/total_ratings por categoría.
- `07_diagnostics.py`: diagnósticos de cobertura y nulos.
- `08_load_postgres.py`: aplica `sql/schema.sql`, carga visitas y POIs a Postgres.

---

## Diagrama del pipeline
```text
      std_2018 (visitas)                POIs (prof/API)          APIs externas
             |                                  |                      |
   01_clean_std / 02_extract_ids                 |              03_fetch_pois
             \___________ ______________________/                      |
                         |                                           /
                04_normalize_pois + 05_label_categories
                         |
                   06_impute_pois
                         |
                 07_diagnostics (QA)
                         |
             data/processed/*.csv/json
                         |
                 08_load_postgres
                         |
                PostgreSQL (Docker)
                         |
   Recommender (CLI) + Routing (OSRM/Geoapify) + Visualización (Folium/HTML)
```

---

## Estructura relevante

- `sql/schema.sql`: esquema Postgres.
- `src/etl/`: scripts ETL (01–08).
- `notebooks/`:
  - `db_load_check.py`: conteos/diagnósticos rápidos en BD.
  - `quick_check.py`: diagnósticos sobre std_clean y POIs enriquecidos.
  - `routing_geoapify_demo.py`: demo de rutas OSRM/Geoapify + Folium.
- `src/recommender/`: esqueleto del recomendador (pendiente de implementación):
  - `cli.py`: entrada por terminal.
  - `scorer.py`: combina scores y re-ranking.
  - `utils_db.py`: helpers de conexión/carga BD.
  - `models/`: content-based (TF-IDF), co-visitas, Markov, embeddings (Word2Vec), ALS/BPR opcional.
  - `features/`: carga de datos, TF-IDF, co-ocurrencias, transiciones, secuencias para Word2Vec.
  - `cache/`: artefactos ligeros (placeholder).
- `data/reports/`: diagnósticos y mapas (`data/reports/maps/*.html`).

---

## Demo de rutas

- Script: `notebooks/routing_geoapify_demo.py`
- Ejemplos:
  - OSRM (gratis) + satélite:  
    `python notebooks/routing_geoapify_demo.py --city osaka --n 2 --mode walk --engine osrm --tiles satellite --open`
  - Geoapify (requiere `GEOAPIFY_API_KEY` en `.env`) + tiles Geoapify:  
    `python notebooks/routing_geoapify_demo.py --city osaka --n 2 --mode walk --engine geoapify --tiles geoapify --open`
- Salida: `data/reports/maps/routing_demo_{city}.html`

---

## Hoja de ruta del recomendador (terminal)

- Features: TF-IDF de categorías, co-ocurrencias, transiciones (Markov), embeddings de secuencia (Word2Vec); opcional ALS/BPR implícito.
- Modelos: content-based, co-visitas, Markov POI/categoría, vecinos en embedding; CF implícito opcional.
- Orquestación: filtrar por ciudad, combinar scores (híbrido), re-ranking por distancia y precio/is_free, selección top-K, ordenación para ruta (vecino más cercano), polilínea OSRM/Geoapify opcional.
- CLI esperado: `python -m src.recommender.cli --user-id ... --city ... --k ... --mode hybrid|content|item|markov --lat/--lon --current-poi --prefs ...`

---

## Backlog breve

- Implementar CLI y scorer del recomendador.
- Construir features (TF-IDF, co-ocurrencias, transiciones, Word2Vec) y baselines.
- Añadir vistas SQL útiles (pois + categorías agregadas).
- (Opcional) FastAPI / capa conversacional (LLM+RAG) tras tener el core en terminal.

---

## Avisos

- `data/raw/` se reserva para datos brutos; no subirlos. Procesados sí están incluidos para arranque inmediato.
