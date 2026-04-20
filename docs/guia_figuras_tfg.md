# Guía de Figuras del TFG

Este documento explica **qué se ve en cada salida visual** de las carpetas:

- `data/reports/figures/tfg/` — 26 figuras de la memoria (fig_01–fig_26 + mermaid)
- `data/reports/figures/dataset/` — 4 figuras de esquema y dataset (ER, ETL, bubble, heatmap)

Incluye PNG/PDF/HTML y el CSV auxiliar de la figura de métricas.

---

## 1) Arquitectura y flujo del sistema

### `fig_01_pipeline_sistema.png` / `fig_01_pipeline_sistema.pdf`
- Qué muestra: diagrama vertical de 5 capas del sistema.
- Capas: Fuentes de datos → ETL → Motores (content, item, markov, embed, als, hybrid, rrf, popular) → Scoring+ruta → API+Frontend.
- Cómo leerla: de arriba hacia abajo, siguiendo flechas entre capas. Resume el flujo completo de datos y decisión.

### `fig_pipeline_mermaid.png`
- Qué muestra: versión Mermaid del pipeline general de producto.
- Cómo leerla: del usuario/frontend hacia backend, motores, reranking y mapa final.

### `fig_scoring_sequence_mermaid.png`
- Qué muestra: diagrama de secuencia de una petición de recomendación.
- Cómo leerla: orden temporal de llamadas `Frontend -> API -> DB -> scorer -> route_planner -> route_builder -> respuesta`.

### `fig_decision_mermaid.png`
- Qué muestra: flujo de decisión por tipo de usuario (con historial/sin historial, con preferencias, etc.).
- Cómo leerla: nodos de decisión que acaban seleccionando familia de motor y salida final.

### `fig_eval_protocol_mermaid.png`
- Qué muestra: versión Mermaid del protocolo de evaluación offline.
- Cómo leerla: split `last_trail_user`, seed POI, comparación top-k vs ground truth, métricas y agregación.

---

## 2) Mapas y distribución espacial

### `fig_02_pois_mapa_categorias.png`
- Qué muestra: mapa de Osaka con todos los POIs disponibles.
- Codificación:
  - Color: categoría amplia (agrupada).
  - Tamaño del punto: rating del POI.
- Cómo leerla: densidad/zonas turísticas y mezcla de categorías en la ciudad.

### `fig_03_heatmap_checkins.html`
- Qué muestra: mapa interactivo Folium con heatmap de check-ins reales en Osaka.
- Cómo leerla: zonas "calientes" = más actividad histórica de usuarios.

### `fig_03_heatmap_checkins.png`
- Qué muestra: versión estática del heatmap (KDE) de check-ins.
- Cómo leerla: intensidad de color = mayor densidad de check-ins.

### `fig_04_hexbin_rating.png`
- Qué muestra: mapa hexbin de Osaka.
- Codificación:
  - Cada hexágono agrupa POIs cercanos.
  - Color = rating medio del hexágono.
- Cómo leerla: calidad media geográfica por zona, suavizando ruido punto a punto.

### `fig_05_tres_ciudades.png`
- Qué muestra: comparación espacial en 3 paneles (Osaka, Istanbul, Petaling Jaya).
- Cómo leerla: diferencias de cobertura/dispersión de POIs entre ciudades.

---

## 3) Markov y secuencias

### `fig_06_markov_heatmap.png`
- Qué muestra: matriz de transición Markov entre categorías amplias.
- Ejes: categoría origen (filas) vs categoría destino (columnas).
- Cómo leerla: valores altos indican transiciones frecuentes entre tipos de POI.

### `fig_07_markov_grafo.png`
- Qué muestra: grafo dirigido de transiciones Markov.
- Codificación:
  - Nodo: categoría.
  - Tamaño nodo: frecuencia relativa.
  - Arista: probabilidad de transición (ancho/color).
- Cómo leerla: estructura global de navegación entre categorías.

### `fig_08_sankey_rutas.html`
- Qué muestra: Sankey interactivo de transiciones en posiciones 1→2→3 de trails (categorías amplias).
- Cómo leerla: grosor del flujo = número de ocurrencias.

### `fig_08_sankey_rutas.png`
- Qué muestra: versión PNG del Sankey (nativa o fallback estático).
- Cómo leerla: igual que la versión HTML, enfocada a uso en memoria/presentación.

### `fig_23_markov_arcos.png`
- Qué muestra: arcos POI→POI (top 40 transiciones) sobre mapa geográfico — **solo Osaka**.
- Codificación: color/grosor del arco = frecuencia de transición (colormap plasma); nodos = POIs origen/destino.
- Fondo oscuro (#1a1a2e) para contraste. Barra de color a la derecha.
- Cómo leerla: "corredores" de movimiento turístico más repetidos en Osaka; los arcos más anchos y claros son las transiciones más frecuentes.

### `fig_24_markov_grafo_mapa.png`
- Qué muestra: grafo Markov geográfico (nodos POI en lat/lon real) — **3 ciudades en subplots**.
- Codificación: color de nodo = categoría amplia (paleta `_CAT_COLORS`); tamaño = nº visitas; aristas = prob ≥ 0.05.
- Leyenda compartida de categorías en la parte inferior.
- Cómo leerla: estructura de movilidad + semántica de categorías en el espacio urbano real de cada ciudad.

### `fig_25_markov_vs_real.png`
- Qué muestra: comparativa 1 fila × 2 cols — **solo Osaka**:
  - Izquierda: transiciones aprendidas por Markov (top 40 arcos, escala Reds).
  - Derecha: rutas reales del dataset (hasta 50 trails ≥ 4 POIs; inicio verde / fin rojo).
- Cómo leerla: grado de alineación entre el patrón aprendido por Markov y el comportamiento real observado en Osaka.

### `fig_26_cold_warm_breakdown.png`
- Qué muestra: comparativa Hit@20 y nDCG@20 para usuarios **cold** (<5 visitas train) vs **warm** (≥5).
- Layout: 2 filas (métricas) × 3 columnas (ciudades); barras agrupadas por motor (hybrid, rrf, item, markov, als, embed).
- Cómo leerla: los motores colaborativos (ALS, Item-Item) penalizan más a usuarios cold; Markov y Content son más robustos ante cold start.

---

## 4) Modelos IA/ML y fusión híbrida

### `fig_09_tsne_embeddings.png`
- Qué muestra: proyección t-SNE de embeddings Word2Vec de POIs (Osaka).
- Codificación:
  - Punto = POI.
  - Color = categoría amplia.
- Cómo leerla: clusters indican similitud semántica/comportamental aprendida por Word2Vec sobre trails.

### `fig_10_als_matriz.png`
- Qué muestra: matriz usuario-POI (top50×top50) usada para explicar ALS.
- Panel izquierdo: matriz binaria de interacciones observadas.
- Panel derecho: texto de interpretación (sparsity y motivación de factorización).
- Cómo leerla: visualiza la dispersión típica del filtrado colaborativo implícito.

### `fig_11_hybrid_weights.png`
- Qué muestra: pesos normalizados del motor híbrido por escenario (`nuevo`, `historial`, `geo`).
- Eje X: motores base (content / item / markov / embed / als).
- Los pesos son específicos por ciudad y están ajustados por tuning (ver `configs/recommender_<qid>.toml`).
- Cómo leerla: qué motor domina en cada contexto operativo y ciudad.

---

## 5) Resultados de evaluación del recomendador

### `fig_12_tabla_metricas.csv`
- Qué contiene: tabla fuente de métricas agregadas por motor y ciudad.
- Columnas: `hit`, `precision`, `recall`, `ndcg`, `cat_hit`, `cat_ndcg`, `novelty`, `diversity`, etc.
- Uso: base tabular para auditoría/reproducibilidad y para regenerar tablas visuales.

### `fig_12_tabla_metricas.png`
- Qué muestra: tabla visual comparativa de motores para las 3 ciudades.
- Resaltado:
  - Verde: mejor valor por columna (excluyendo random donde aplica).
  - Rosa: peor valor por columna (excluyendo random donde aplica).
- Cómo leerla: comparación directa de rendimiento global por motor.

### `fig_13_barras_agrupadas.png`
- Qué muestra: barras agrupadas de **nDCG@20** por motor y ciudad.
- Línea roja discontinua: baseline `random` medio (si existe en resultados).
- Cómo leerla: distancia de cada motor frente al baseline trivial; hybrid e item-item son consistentemente superiores.

### `fig_14_radar_chart.png`
- Qué muestra: radar multi-métrica de motores seleccionados.
- Métricas: hit, precision, recall, ndcg, novelty, diversity.
- Cómo leerla: perfil equilibrado/fuerte-débil de cada motor en una sola vista.

### `fig_14b_heatmap_metricas.png`
- Qué muestra: heatmap completo motor × métrica (media 3 ciudades), ordenado por hit.
- Cómo leerla: visión compacta de dominancia relativa por métrica.

### `fig_15_curvas_metricas_k.png`
- Qué muestra: barras por motor en Osaka para `Precision@20`, `Recall@20`, `nDCG@20`.
- Cómo leerla: comparación directa de trade-off de ranking a k fijo.

### `fig_20_eval_protocolo.png`
- Qué muestra: infografía del protocolo de evaluación offline end-to-end.
- Contenido clave: split `last_trail_user`, flag `--fair`, 9 motores, seed POI, ground truth, métricas ranking + ruta, segmentación cold/warm.
- Cómo leerla: documento de metodología evaluativa (ideal para capítulo experimental).

### `fig_21_comparativa_literatura.png`
- Qué muestra: posicionamiento del TFG frente a resultados reportados en literatura (NDCG@10 aproximado).
- Referencias incluidas: BPR-MF, Markov puro, FPMC, Item-KNN (ML tradicional); GRU4Rec, GETNext, STHGCN (Deep Learning SOTA).
- Nota: comparación orientativa — el TFG usa NDCG@20 y protocolo `last_trail_user`, la literatura usa NDCG@10 y leave-one-out.
- Cómo leerla: los motores del TFG superan todos los baselines ML tradicionales; el mejor (Item-Item, 0.172) iguala GRU4Rec.

### `fig_22_comparativa_hit.png`
- Qué muestra: comparativa específica de **Hit@K** entre resultados del TFG y referencias de literatura.
- Cómo leerla: dónde se sitúan los motores del TFG respecto a baselines y estado del arte reportado.

---

## 6) Dataset y comportamiento de usuarios

### `fig_16_distribucion_categorias.png`
- Qué muestra: distribución por categoría amplia en Osaka.
- Barras:
  - `POIs`: inventario disponible.
  - `check-ins/10`: uso real histórico escalado.
- Cómo leerla: diferencia entre oferta de POIs y demanda real; Food domina el uso pero no el inventario.

### `fig_17_long_tail_usuarios.png`
- Qué muestra:
  - Izquierda: histograma log de actividad por usuario.
  - Derecha: curva de Lorenz.
- Cómo leerla: evidencia del efecto long-tail y concentración de actividad en pocos usuarios muy activos.

### `fig_18_heatmap_temporal.png`
- Qué muestra: patrón temporal hora × día de semana de check-ins (Osaka).
- Cómo leerla: franjas horarias/días con mayor intensidad de actividad; pico en horas de comida y fin de semana.

### `fig_19_longitud_trails.png`
- Qué muestra: histograma de longitud de trails (#POIs por trail).
- Líneas: media y mediana.
- Cómo leerla: tamaño típico de ruta real en el dataset; mayoría de trails tienen 2–5 POIs.

---

## 7) Figuras de dataset y esquema (`data/reports/figures/dataset/`)

Generadas por `scripts/generate_dataset_figures.py`. Carpeta independiente de las figuras de la memoria principal.

### `fig_er_diagram.png`
- Qué muestra: diagrama entidad-relación del esquema PostgreSQL con las 4 tablas principales.
- Tablas: `visits`, `pois`, `poi_categories`, `saved_routes`.
- Relaciones: visits N:1 pois (via venue_id → fsq_id); pois 1:N poi_categories (via fsq_id); pois → saved_routes (conceptual via city_qid, dashed).
- Codificación: cabecera de tabla en color por tabla; PK en verde, FK en morado; filas alternas sombreadas.
- Cómo leerla: estructura de datos subyacente del sistema; punto de partida para el capítulo de diseño de BD.

### `fig_etl_flow.png`
- Qué muestra: pipeline ETL completo desde el CSV Foursquare crudo hasta las tablas PostgreSQL.
- Flujo: `Foursquare CSV` → `01_clean_std` → `02_extract_ids` → `03_fetch_pois` → `04_normalize_pois` → `05_label_categories` → `06_impute_pois` → `07_diagnostics` → `08_load_postgres` → `PostgreSQL DB`.
- A la derecha de cada script: fichero de salida intermedio (std_clean.csv, venue_ids.txt, pois_raw.json, etc.).
- Cómo leerla: de arriba hacia abajo; cada caja es un script ETL, las etiquetas laterales son los artefactos que produce.

### `fig_bubble_dataset.png`
- Qué muestra: 3 ciudades como burbujas cuyo tamaño es proporcional a √(check-ins); dentro de cada burbuja, un donut con la distribución de categorías amplias.
- Osaka (~200K check-ins) es visiblemente mayor que Istanbul (~40K) y Petaling Jaya (~35K).
- Leyenda compartida de categorías en la parte inferior.
- Cómo leerla: permite comparar volumen relativo y composición temática de cada ciudad de un solo vistazo.

### `fig_heatmap_coverage.png`
- Qué muestra: matriz ciudad × 6 estadísticas del dataset (usuarios únicos, POIs, check-ins, trails, media POIs/trail, sparsidad).
- Coloreado por intensidad relativa dentro de cada columna (normalización 0–1 por métrica).
- Cada celda muestra el valor real.
- Cómo leerla: permite ver de un vistazo por qué Istanbul es más difícil (menos datos, más sparsidad) que Osaka; explica las diferencias de rendimiento entre ciudades.

---

## Nota de interpretación técnica

- Algunas figuras con basemap (`fig_02`, `fig_03`, `fig_04`, `fig_05`, `fig_23`, `fig_24`, `fig_25`) dependen de acceso a proveedores de tiles web (CartoDB, OSM).
- Si hay bloqueo de red/firewall en el entorno Python, se puede ver fondo neutro en vez de mapa base. Esto **no invalida los datos de la figura**, solo afecta la capa visual de contexto cartográfico.
- Las figuras `fig_23` y `fig_25` muestran solo **Osaka** (mayor volumen de datos, resultados más representativos). `fig_24` mantiene las 3 ciudades en subplots.
- Los resultados de evaluación recogidos en fig_12–fig_22 y fig_26 corresponden al benchmark con protocolo `last_trail_user --fair`, seed=42, max_users=300, k=20.
