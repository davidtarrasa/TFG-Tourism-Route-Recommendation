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
