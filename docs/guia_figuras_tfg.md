# Guía de Figuras del TFG

Este documento explica **qué se ve en cada salida visual** de la carpeta:

- `data/reports/figures/tfg/`

Incluye PNG/PDF/HTML y el CSV auxiliar de la figura de métricas.

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

## 2) Mapas y distribución espacial

### `fig_02_pois_mapa_categorias.png`
- Qué muestra: mapa de Osaka con todos los POIs disponibles.
- Codificación:
  - Color: categoría amplia (agrupada).
  - Tamaño del punto: rating del POI.
- Cómo leerla: densidad/zonas turísticas y mezcla de categorías en la ciudad.

### `fig_03_heatmap_checkins.html`
- Qué muestra: mapa interactivo Folium con heatmap de check-ins reales en Osaka.
- Cómo leerla: zonas “calientes” = más actividad histórica de usuarios.

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
- Qué muestra: arcos POI→POI (top transiciones) sobre mapa.
- Codificación:
  - Grosor/color del arco = frecuencia de transición.
  - Nodos destacados = POIs más visitados.
- Cómo leerla: “carreteras” de movimiento más repetidas en comportamiento real.

### `fig_24_markov_grafo_mapa.png`
- Qué muestra: grafo Markov geográfico (nodos POI con coordenadas reales).
- Codificación:
  - Nodos por categoría amplia.
  - Aristas con probabilidad mínima (filtrado) para evitar ruido.
- Cómo leerla: estructura de movilidad + semántica de categorías en el espacio urbano.

### `fig_25_markov_vs_real.png`
- Qué muestra: comparativa lado a lado.
  - Izquierda: transiciones aprendidas por Markov.
  - Derecha: rutas reales del dataset.
- Cómo leerla: grado de alineación entre patrón aprendido y comportamiento observado.

## 4) Modelos IA/ML y fusión híbrida

### `fig_09_tsne_embeddings.png`
- Qué muestra: proyección t-SNE de embeddings Word2Vec de POIs (Osaka).
- Codificación:
  - Punto = POI.
  - Color = categoría amplia.
- Cómo leerla: clusters indican similitud semántica/comportamental aprendida.

### `fig_10_als_matriz.png`
- Qué muestra: matriz usuario-POI (top50×top50) usada para explicar ALS.
- Panel izquierdo: matriz binaria de interacciones observadas.
- Panel derecho: texto de interpretación (sparsity y motivación de factorización).
- Cómo leerla: visualiza la dispersión típica del filtrado colaborativo.

### `fig_11_hybrid_weights.png`
- Qué muestra: pesos normalizados del híbrido por escenario (`nuevo`, `historial`, `geo`).
- Eje X: motores base (content/item/markov/embed/als).
- Cómo leerla: qué motor domina en cada contexto operativo.

## 5) Resultados de evaluación del recomendador

### `fig_12_tabla_metricas.csv`
- Qué contiene: tabla fuente de métricas agregadas por motor y ciudad.
- Columnas típicas: `hit`, `precision`, `recall`, `ndcg`, `novelty`, `diversity`, etc.
- Uso: base tabular para auditoría/reproducibilidad y para regenerar tablas visuales.

### `fig_12_tabla_metricas.png`
- Qué muestra: tabla visual comparativa de motores para las 3 ciudades.
- Resaltado:
  - Verde: mejor valor por columna (excluyendo random donde aplica).
  - Rosa: peor valor por columna (excluyendo random donde aplica).
- Cómo leerla: comparación directa de rendimiento global por motor.

### `fig_13_barras_agrupadas.png`
- Qué muestra: barras agrupadas de **nDCG** por motor y ciudad.
- Línea roja discontinua: baseline `random` medio (si existe en resultados).
- Cómo leerla: distancia de cada motor frente al baseline trivial.

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
- Contenido clave: split `last_trail_user`, `--fair`, 9 motores, seed, ground truth, métricas, cold/warm.
- Cómo leerla: documento de metodología evaluativa (ideal para capítulo experimental).

### `fig_20_comparativa_literatura.png`
- Qué muestra: posicionamiento del TFG frente a resultados reportados en literatura.
- Cómo leerla: referencia contextual (comparación orientativa con cautelas de protocolo).

### `fig_22_comparativa_hit.png`
- Qué muestra: comparativa específica de **Hit@K** entre resultados del TFG y referencias de literatura.
- Cómo leerla: dónde se sitúan los motores del TFG respecto a baseline/estado del arte reportado.

## 6) Dataset y comportamiento de usuarios

### `fig_16_distribucion_categorias.png`
- Qué muestra: distribución por categoría amplia en Osaka.
- Barras:
  - `POIs`: inventario disponible.
  - `check-ins/10`: uso real histórico escalado.
- Cómo leerla: diferencia entre oferta de POIs y demanda real.

### `fig_17_long_tail_usuarios.png`
- Qué muestra:
  - Izquierda: histograma log de actividad por usuario.
  - Derecha: curva de Lorenz.
- Cómo leerla: evidencia del efecto long-tail y concentración de actividad.

### `fig_18_heatmap_temporal.png`
- Qué muestra: patrón temporal hora × día de semana de check-ins (Osaka).
- Cómo leerla: franjas horarias/días con mayor intensidad de actividad.

### `fig_19_longitud_trails.png`
- Qué muestra: histograma de longitud de trails (#POIs por trail).
- Líneas: media y mediana.
- Cómo leerla: tamaño típico de ruta real en el dataset.

---

## Nota de interpretación técnica

- Algunas figuras con basemap (`fig_02`, `fig_03`, `fig_04`, `fig_05`, `fig_23`, `fig_24`, `fig_25`) dependen de acceso a proveedores de tiles web.
- Si hay bloqueo de red/firewall en el entorno Python, se puede ver fondo neutro en vez de mapa base.
- Esto **no invalida los datos de la figura**, solo afecta la capa visual de contexto cartográfico.
