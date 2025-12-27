# TFG-Tourism-Route-Recommendation

Final Degree Project: Tourism route recommendation and visualization system. Includes data processing, ML-based recommendations, interactive map visualization, and conversational AI (LLMs + RAG) for natural language user interaction.

# TFG – Sistema de recomendación y visualización de rutas turísticas basado en IA

Este Trabajo de Fin de Grado desarrolla una aplicación capaz de **recomendar y visualizar rutas turísticas personalizadas** a partir de datos reales de usuarios.

El sistema combina:

- **ML clásico** (recomendación y clustering),
- **optimización de rutas**,
- una **base de datos relacional (PostgreSQL)** para los datos estructurados,
- y, en fases posteriores, **LLMs + RAG** y una **interfaz web con mapas**. :contentReference[oaicite:0]{index=0}

---

## 🧾 Objetivos

- Procesar y estructurar un dataset de rutas reales (Semantic Trails std_2018 + venues de Foursquare).
- Diseñar e implementar un sistema de recomendación:
  - Content-based filtering.
  - Collaborative filtering (usuarios con historial).
  - Clustering de POIs.
- Optimizar rutas turísticas teniendo en cuenta distancias y horarios.
- Almacenar los datos limpios en una **base de datos PostgreSQL** accesible desde los scripts del proyecto.
- Exponer la lógica en un backend (FastAPI) con endpoints REST.
- Añadir una capa conversacional con LLM + RAG para entrada en lenguaje natural.
- Desarrollar una interfaz web para visualizar rutas sobre un mapa y conversar con el sistema.

---

## Arranque rápido (Docker + Postgres)

1. Copia `.env.example` a `.env` y rellena `FOURSQUARE_API_KEY` si vas a usar la API.
2. Levanta Postgres (mapeado al puerto host 55432) y pgAdmin:  
   `docker compose up -d db pgadmin`
3. Prepara el entorno Python para cargar datos:  
   `python -m venv .venv && .\.venv\Scripts\activate`  
   `pip install -r requirements.txt`
4. Carga el esquema y los datos procesados:  
   `python src/etl/08_load_postgres.py --dsn postgresql://tfg:tfgpass@localhost:55432/tfg_routes`
5. Verifica conteos:  
   `docker compose exec -T db psql -U tfg -d tfg_routes -c "SELECT COUNT(*) FROM visits; SELECT COUNT(*) FROM pois; SELECT COUNT(*) FROM poi_categories;"`
6. pgAdmin: `http://localhost:8080` (creds en `.env`). Registra un server con host `db`, port `5432`, database `tfg_routes`, user `tfg`, pass `tfgpass`.

Notas:
- El puerto host es 55432 porque muchos equipos ya tienen Postgres en 5432 (`55432:5432` en `docker-compose.yml`).
- Los datos procesados (`data/processed/std_clean.csv` y `pois_enriched_*.json`) ya vienen en el repo, así que la carga es directa.

---

## 🏗️ Arquitectura general

La arquitectura objetivo del proyecto es:

```text
Usuario (terminal / web / chat)
              ↓
        Frontend web
   (Streamlit o React + Leaflet)
              ↓
          Backend API
             FastAPI
              ↓
 ┌─────────────────────────────────────┐
 │  Lógica de negocio y recomendador  │
 │  - ML clásico (content-based,      │
 │    collaborative, clustering)      │
 │  - Optimización de rutas           │
 └─────────────────────────────────────┘
          ↓                 ↓
  Base de datos PostgreSQL   Motor RAG + LLM
 (POIs, trails, usuarios)   (embeddings, búsqueda)
```

---

## Ideas de mejora / backlog

- Recomendadores básicos adicionales:
  - POI más cercano: matriz de distancias entre POIs por ciudad; dado un POI actual, recomendar el más próximo (ojo al tamaño de la matriz en Osaka/PJ).
  - Markov por POI: matriz de transiciones entre POIs a partir de rutas históricas; dado el POI actual, elegir el siguiente con máxima probabilidad de transición.
  - Markov por categoría: transiciones entre categorías; si hay empates, romper con el POI más cercano.
- Añadir vistas SQL útiles (ej. pois con categorías agregadas) para consultas rápidas desde API/notebooks.
- Baselines de recomendación en terminal (content-based / colaborative / clustering) con un CLI unificado.
- Esqueleto FastAPI con `/health` y `/recommend` leyendo de Postgres para conectar pronto con frontend.
