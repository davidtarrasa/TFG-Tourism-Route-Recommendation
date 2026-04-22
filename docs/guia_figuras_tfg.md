# Guía de Figuras del TFG

Este documento explica **qué se ve en cada salida visual** de las carpetas:

- `data/reports/figures/tfg/` — 26 figuras de la memoria (fig_01–fig_26 + mermaid)
- `data/reports/figures/dataset/` — 4 figuras de esquema y dataset (ER, ETL, bubble, heatmap)

Incluye PNG/PDF/HTML y el CSV auxiliar de la figura de métricas. Para cada figura se incluye tanto la descripción técnica como la interpretación en lenguaje natural.

---

## 1) Arquitectura y flujo del sistema

### `fig_01_pipeline_sistema.png` / `fig_01_pipeline_sistema.pdf`

**Qué muestra:** diagrama vertical de 5 capas del sistema.

**Cómo interpretarla:** Imagina que el sistema es una fábrica en 5 pisos:

- **Piso 1 (azul) — Fuentes de datos:** de dónde viene la información: datos históricos de Foursquare (la gente que hizo check-in en sitios), información de los POIs, y la API para calcular rutas.
- **Piso 2 (rosa) — ETL:** la "cocina" donde se limpian y preparan esos datos antes de usarlos.
- **Piso 3 (amarillo) — Motores:** el corazón. Son 8 algoritmos diferentes que cada uno tiene su forma de recomendar. Como 8 asesores distintos que dan su opinión.
- **Piso 4 (rojo) — Scoring + Ruta:** los 8 asesores dan sus listas, aquí se fusionan, se puntúan y se ordenan los POIs en una ruta geográfica lógica.
- **Piso 5 (gris) — API + Frontend:** lo que ve el usuario final: la web con el mapa.

La flecha va siempre hacia abajo. Así funciona el sistema de principio a fin.

---

### `fig_pipeline_mermaid.png`

**Qué muestra:** versión Mermaid del pipeline general de producto.

**Cómo interpretarla:** sigue el camino de una petición desde que el usuario abre la web hasta que recibe su ruta en el mapa. Del frontend hacia el backend, los motores de recomendación, el reranking, y finalmente el mapa con la ruta trazada. Es la versión "funcional" del fig_01, centrada en el flujo de una petición concreta en lugar de en los componentes del sistema.

---

### `fig_scoring_sequence_mermaid.png`

**Qué muestra:** diagrama de secuencia de una petición de recomendación.

**Cómo interpretarla:** orden temporal de llamadas: `Frontend → API → DB → scorer → route_planner → route_builder → respuesta`. Cada flecha horizontal representa una llamada entre componentes y el tiempo transcurre de arriba hacia abajo. Es el diagrama más útil para entender la latencia del sistema: dónde tarda más y qué partes son secuenciales vs. paralelas.

---

### `fig_decision_mermaid.png`

**Qué muestra:** flujo de decisión por tipo de usuario (con historial/sin historial, con preferencias, etc.).

**Cómo interpretarla:** árbol de decisión que empieza en "¿qué tipo de usuario es?" y va ramificándose. Cada nodo de decisión representa una comprobación (¿tiene historial? ¿hay preferencias de categoría? ¿viene petición con ubicación geográfica?) y acaba seleccionando qué familia de motor usar y qué parámetros ajustar. Visualiza el "cerebro de la puerta de entrada" del sistema.

---

### `fig_eval_protocol_mermaid.png`

**Qué muestra:** versión Mermaid del protocolo de evaluación offline.

**Cómo interpretarla:** versión esquemática del mismo proceso explicado en detalle en fig_20: split `last_trail_user`, seed POI, comparación top-k vs. ground truth, métricas y agregación. Sirve como diagrama de referencia rápida para el capítulo de evaluación experimental.

---

## 2) Mapas y distribución espacial

### `fig_02_pois_mapa_categorias.png`

**Qué muestra:** mapa de Osaka con todos los POIs disponibles del dataset.

**Codificación:** color = categoría amplia (Food, Transport, Shopping, etc.); tamaño del punto = rating del POI.

**Cómo interpretarla:** la figura responde a "¿dónde están los lugares y cómo se agrupan en la ciudad?". Los puntos más grandes son los POIs mejor valorados. Las zonas con mayor densidad de puntos son los núcleos turísticos de Osaka. La mezcla de colores en esas zonas densas indica que los turistas encuentran variedad de categorías en los mismos barrios, lo que justifica que las rutas recomendadas sean variadas temáticamente.

**Qué se lee de la figura:** hay zonas claras de alta concentración turística (centro de Osaka) y zonas periféricas más dispersas. Los POIs de Food (categoría dominante) están distribuidos por toda la ciudad, mientras que los de Transport se concentran en nudos específicos.

---

### `fig_03_heatmap_checkins.html`

**Qué muestra:** mapa interactivo Folium con heatmap de check-ins reales en Osaka.

**Cómo interpretarla:** cada punto de calor es un check-in histórico de un usuario real. Las zonas más rojas/brillantes son donde más turistas han ido de verdad. A diferencia del fig_02 que muestra dónde están los POIs (oferta), este mapa muestra dónde va la gente de verdad (demanda). La versión HTML permite hacer zoom y ver la distribución en detalle.

---

### `fig_03_heatmap_checkins.png`

**Qué muestra:** versión estática del heatmap (KDE) de check-ins.

**Cómo interpretarla:** igual que la versión HTML pero en formato estático apto para la memoria. La intensidad de color refleja densidad de check-ins: los focos más brillantes son los puntos de mayor atracción turística real, independientemente de cuántos POIs hay en esa zona. Si un área tiene muchos POIs pero poca densidad de check-ins, significa que esos POIs no atraen mucho en la práctica.

---

### `fig_04_hexbin_rating.png`

**Qué muestra:** mapa hexbin de Osaka donde cada hexágono agrupa POIs cercanos.

**Codificación:** color del hexágono = rating medio de los POIs que contiene.

**Cómo interpretarla:** suaviza el ruido punto a punto y permite ver la calidad geográfica media por zona. Una zona con hexágono verde/amarillo oscuro tiene una concentración de POIs bien valorados; una zona azul/pálida tiene POIs menos valorados o muy pocos datos. Es útil para detectar "barrios de calidad" para el turista.

**Qué se lee de la figura:** el centro de Osaka tiende a tener ratings más altos (mayor concentración de lugares consolidados) frente a zonas periféricas donde los datos son más escasos y los ratings más dispersos.

---

### `fig_05_tres_ciudades.png`

**Qué muestra:** comparación espacial en 3 paneles (Osaka, Istanbul, Petaling Jaya).

**Cómo interpretarla:** permite ver de un vistazo cómo son las tres ciudades que usa el TFG. Osaka aparece como la ciudad con más cobertura y densidad de POIs. Istanbul tiene una distribución concentrada en pocas zonas. Petaling Jaya muestra una distribución más dispersa. Estas diferencias de cobertura y dispersión geográfica afectan directamente al rendimiento de los motores: cuantos más datos y más densos, mejores recomendaciones.

---

## 3) Markov y secuencias

### `fig_06_markov_heatmap.png`

**Qué muestra:** matriz de transición Markov entre categorías amplias.

**Ejes:** categoría origen (filas) vs. categoría destino (columnas). Cada celda es la probabilidad de ir de esa categoría a la otra.

**Cómo interpretarla:** es el "patrón estadístico" de cómo se mueven los turistas entre tipos de lugar. Cuanto más oscura la celda, más probable es esa transición. La diagonal principal (quedarse en la misma categoría) suele ser prominente para Food, lo que indica que los turistas visitan varios restaurantes seguidos. Las celdas fuera de diagonal revelan "saltos" entre categorías frecuentes.

**Qué se lee de la figura:** Food → Food es la transición más frecuente. Transport aparece como nodo intermediario frecuente (la gente pasa por transporte para llegar a otros sitios). Shopping → Food y Transport → Food son transiciones habituales, lo que tiene mucho sentido turísticamente.

---

### `fig_07_markov_grafo.png`

**Qué muestra:** grafo dirigido de transiciones Markov entre categorías.

**Codificación:** nodo = categoría; tamaño del nodo = frecuencia relativa de esa categoría; arista dirigida = probabilidad de transición; ancho y color de arista = magnitud de esa probabilidad.

**Cómo interpretarla:** es la representación visual del mismo modelo que fig_06 pero como red. Permite ver qué categorías son "hubs" (conectadas con muchas otras), qué categorías son receptoras (muchas flechas entrantes) y cuáles son emisoras (muchas flechas salientes). Food y Transport suelen aparecer como los nodos centrales del grafo.

---

### `fig_08_sankey_rutas.html` / `fig_08_sankey_rutas.png`

**Qué muestra:** Sankey de transiciones en posiciones 1→2→3 de los trails, por categorías amplias.

**Cómo interpretarla:** esta figura responde a "cuando alguien va a un sitio de transporte, ¿adónde suele ir después?".

- Cada columna es una posición de la ruta: **1.º POI → 2.º POI → 3.º POI**. Las categorías (Food, Transport, Shopping...) aparecen en las tres columnas.
- Las bandas que conectan una columna con la siguiente representan cuánta gente hizo esa transición: cuanto más gruesa la banda, más gente lo hizo.

**Qué se lee de la figura:** Transport y Food son las categorías más comunes al principio y al final de las rutas. La mayoría de flujos van de Transport a Food o a Shopping, lo cual tiene mucho sentido turísticamente: la gente llega a un punto de transporte, y de ahí va a comer o a comprar. La versión HTML permite hacer hover sobre cada banda para ver el número exacto de ocurrencias.

---

### `fig_23_markov_arcos.png`

**Qué muestra:** arcos POI→POI (top 40 transiciones más frecuentes) sobre mapa geográfico. **Solo Osaka.**

**Codificación:** color y grosor del arco = frecuencia de transición (colormap plasma, de oscuro a claro/amarillo); nodos = POIs origen/destino; fondo oscuro (#1a1a2e) para contraste; barra de color a la derecha.

**Cómo interpretarla:** es la versión geográfica real del modelo Markov. En lugar de ver categorías abstractas (fig_06/07), aquí se ven las transiciones entre lugares concretos sobre el mapa real de Osaka. Los arcos más anchos y brillantes son las rutas entre pares de POIs que más turistas han recorrido históricamente.

**Qué se lee de la figura:** emergen "corredores" de movimiento turístico: conexiones entre POIs que aparecen repetidamente en los trails reales. Estos corredores son la base del motor Markov: si vas al POI A, el sistema te recomendará los POIs a los que más gente suele ir desde A.

---

### `fig_24_markov_grafo_mapa.png`

**Qué muestra:** grafo Markov geográfico con nodos en sus coordenadas lat/lon reales. **3 ciudades en subplots.**

**Codificación:** color de nodo = categoría amplia (paleta `_CAT_COLORS`); tamaño del nodo = número de visitas; aristas = transiciones con probabilidad ≥ 0.05; leyenda compartida de categorías en la parte inferior.

**Cómo interpretarla:** combina la semántica de categorías (color) con la localización real (lat/lon) y la intensidad de uso (tamaño). Permite ver simultáneamente dónde está cada lugar, qué tipo de lugar es y cuánto se visita, y qué trayectos son habituales. Comparando los 3 subplots se aprecia directamente por qué Osaka tiene más datos que Istanbul y Petaling Jaya.

---

### `fig_25_markov_vs_real.png`

**Qué muestra:** comparativa 2 paneles — **solo Osaka**.

- **Izquierda:** transiciones aprendidas por Markov (top 40 arcos, escala Reds).
- **Derecha:** rutas reales del dataset (hasta 50 trails de ≥ 4 POIs; inicio marcado en verde, fin en rojo).

**Cómo interpretarla:** es el "examen de realismo" del motor Markov. Si el modelo aprendió bien, los arcos del panel izquierdo deberían coincidir geográficamente con los flujos visibles en el panel derecho. Las zonas donde coinciden validan el aprendizaje; las zonas donde difieren revelan rutas reales que el modelo no captura bien (posiblemente trails con poca frecuencia).

---

## 4) Modelos IA/ML y fusión híbrida

### `fig_09_tsne_embeddings.png`

**Qué muestra:** proyección t-SNE de los embeddings Word2Vec de los POIs de Osaka.

**Codificación:** cada punto es un POI; color = categoría amplia.

**Cómo interpretarla:** los puntos no están colocados por su posición geográfica en la ciudad, sino por **similitud de comportamiento**: dos POIs están cerca en la figura si los usuarios los visitan en contextos parecidos.

El algoritmo Word2Vec "aprende" que si mucha gente va primero al POI A y luego al POI B, esos dos son similares. Después t-SNE proyecta esas similitudes en 2D para que se puedan visualizar.

**Qué se lee de la figura:** una nube bastante mezclada de colores, lo que indica que los POIs de distintas categorías no forman grupos perfectamente separados. Esto significa que el comportamiento de los usuarios es variado y complejo: la gente no solo visita cada tipo de lugar con el mismo tipo. Si vieras grupos muy compactos y separados por color, significaría que la gente solo visita cada categoría con la misma categoría — pero no es el caso. Esta mezcla justifica que el motor Word2Vec (Embed) pueda hacer recomendaciones inter-categoría relevantes.

---

### `fig_10_als_matriz.png`

**Qué muestra:** matriz usuario-POI (top 50 × top 50) usada para explicar el motor ALS.

**Paneles:** izquierda = matriz binaria de interacciones observadas (azul oscuro = visitó, blanco = no visitó); derecha = texto de interpretación con la cifra de sparsity y motivación de la factorización matricial.

**Cómo interpretarla:** la cuadrícula de la izquierda es una tabla: filas = 50 usuarios, columnas = 50 POIs. Cada cuadrado azul oscuro significa "este usuario visitó este POI". Los cuadrados blancos = no visitó.

Lo que salta a la vista: hay muy pocos cuadrados azules — el **74% de la tabla está en blanco**. Esto es normal: ningún turista visita todos los POIs de una ciudad. El problema es que si solo sabemos qué visitó cada uno, no podemos recomendar bien a alguien que no visitó casi nada.

**Qué resuelve ALS:** "rellena" los blancos matemáticamente, estimando qué le gustaría a cada usuario basándose en patrones similares entre usuarios y POIs. Si el usuario A y el usuario B visitaron los mismos 3 POIs, y B también visitó el POI X, ALS asume que A probablemente también le gustaría X.

---

### `fig_11_hybrid_weights.png`

**Qué muestra:** pesos normalizados del motor híbrido por escenario (`nuevo`, `historial`, `geo`).

**Codificación:** eje X = motores base (content / item / markov / embed / als); las 3 barras por motor son los 3 escenarios posibles; los pesos son específicos por ciudad y están ajustados por tuning (ver `configs/recommender_<qid>.toml`).

**Cómo interpretarla:** el motor híbrido no usa siempre los mismos 5 motores con el mismo peso. Cambia según el contexto. Esta figura muestra exactamente eso:

- **Azul (nuevo):** usuario que no tiene historial. El sistema no sabe nada de él.
- **Naranja (historial):** usuario que ya ha visitado cosas antes.
- **Rojo (geo):** petición basada en ubicación geográfica actual.

**Qué se lee de la figura:**
- Cuando el usuario es nuevo (azul), **Content domina con ~0.6** — tiene sentido porque sin historial, lo único que puedes hacer es recomendar por características del POI.
- Cuando tiene historial (naranja), Content baja mucho y **suben Item-Item, Markov y ALS** — porque ahora sí tienes datos del usuario para los motores colaborativos.
- En modo geo (rojo), **Markov y ALS suben al máximo (~0.35 cada uno)** porque la ubicación y los patrones de secuencia son lo más relevante cuando la petición viene condicionada por una ubicación concreta.

---

## 5) Resultados de evaluación del recomendador

### `fig_12_tabla_metricas.csv`

**Qué contiene:** tabla fuente de métricas agregadas por motor y ciudad.

**Columnas:** `hit`, `precision`, `recall`, `ndcg`, `cat_hit`, `cat_ndcg`, `novelty`, `diversity`, etc.

**Uso:** base tabular para auditoría/reproducibilidad y para regenerar tablas visuales. Si quieres el número exacto de cualquier métrica para cualquier motor y ciudad, este es el fichero de referencia.

---

### `fig_12_tabla_metricas.png`

**Qué muestra:** tabla visual comparativa de motores para las 3 ciudades, con celdas coloreadas.

**Codificación:** verde = mejor valor por columna (excluyendo random donde aplica); rosa = peor valor por columna.

**Cómo interpretarla:** es el "cuadro de honor" del TFG. De un vistazo permite identificar qué motor es el mejor en cada métrica y ciudad, y cuál es el peor. El color verde hace que salte a la vista que Hybrid e Item-Item dominan las métricas de ranking (hit, ndcg) mientras que otros motores como Popular o Content tienen ventajas en novelty/diversity.

---

### `fig_13_barras_agrupadas.png`

**Qué muestra:** barras agrupadas de nDCG@20 por motor y ciudad.

**Codificación:** cada grupo de barras es un motor; los colores de las barras son las ciudades; línea roja discontinua = baseline `random` medio.

**Cómo interpretarla:** el nDCG mide no solo si el sistema acertó, sino si lo puso en una posición alta de la lista. La línea roja es el suelo: cualquier motor que esté por debajo del random es peor que no recomendar nada. La distancia de cada barra por encima de la línea roja indica cuánto valor aporta ese motor respecto al azar.

**Qué se lee de la figura:** hybrid e item-item son consistentemente superiores al resto. Istanbul (menor volumen de datos) tiende a tener barras más bajas que Osaka y Petaling Jaya para casi todos los motores.

---

### `fig_14_radar_chart.png`

**Qué muestra:** radar multi-métrica de motores seleccionados (rrf, markov, hybrid, item, popular, random).

**Métricas en el radar:** hit, precision, recall, ndcg, novelty, diversity.

**Nota importante:** la media se calcula **solo sobre Osaka + Petaling Jaya** (Istanbul excluida por volumen ~4× inferior). ALS, Embed y Content omitidos por claridad visual; ver fig_14b para todos los motores.

**Cómo interpretarla:** cada motor aparece como un polígono. Un polígono más grande y equilibrado es mejor. Permite ver de un vistazo el "perfil" de cada motor: hybrid tiene un polígono grande pero algo irregular; popular tiene un perfil muy desequilibrado (alta diversity, bajo recall); random forma el polígono más pequeño.

**Qué se lee de la figura:** no existe un motor perfecto en todas las dimensiones. Hybrid maximiza el área total del polígono, pero Markov tiene ventajas específicas en novelty. El trade-off entre calidad de ranking (hit/ndcg) y variedad (diversity/novelty) es visible directamente en los polígonos.

---

### `fig_14b_heatmap_metricas.png`

**Qué muestra:** heatmap completo motor × métrica, ordenado por hit descendente.

**Nota:** media calculada sobre **Osaka + Petaling Jaya** (Istanbul excluida por volumen ~4× inferior, igual que fig_14 y fig_21/22).

**Cómo interpretarla:** visión compacta de todos los motores y todas las métricas en una sola tabla coloreada. La ordenación por hit descendente pone los motores más fuertes arriba. Las celdas más oscuras son los valores más altos. Permite identificar de un vistazo qué métricas son "fáciles" para todos (hit) vs. cuáles discriminan más entre motores (precision, ndcg).

---

### `fig_15_curvas_metricas_k.png`

**Qué muestra:** barras por motor en Osaka para `Precision@20`, `Recall@20`, `nDCG@20`.

**Cómo interpretarla:** los tres paneles permiten comparar el trade-off de ranking a k fijo (k=20). Precision mide cuántos de los 20 recomendados son relevantes; Recall mide qué fracción de lo relevante fue capturado; nDCG mide ambas cosas pero ponderando más los aciertos en posiciones altas. Un motor puede tener Precision baja pero Recall alta si recomienda cosas relevantes pero dispersas en la lista.

---

### `fig_20_eval_protocolo.png`

**Qué muestra:** infografía del protocolo de evaluación offline end-to-end.

**Cómo interpretarla:** es el "reglamento del examen" para medir si los motores recomiendan bien o mal. Tiene 5 pasos:

1. **Partición:** se separan los datos. Los últimos trails de cada usuario van al "test" (son las respuestas correctas que el sistema no puede ver). El resto es para entrenar.

2. **Reentrenamiento justo:** cada motor se entrena solo con los datos de entrenamiento, nunca ve el test. Esto simula el entorno real donde el sistema no conoce el futuro.

3. **Bucle de evaluación:** para cada usuario de test, se le da al motor un POI de inicio (seed) y se le pide que recomiende los siguientes 20. Luego se compara con lo que el usuario hizo de verdad.

4. **Métricas:** se mide si acertó (Hit), en qué posición acertó (nDCG), qué tan variado fue (Diversity), etc.

5. **Resultados:** se hace para las 3 ciudades, los 9 motores y se segmenta por usuarios nuevos (cold) vs. activos (warm).

El flag `--fair` garantiza que cada motor use exactamente el mismo conjunto de usuarios y seeds, haciendo la comparación completamente justa.

---

### `fig_21_comparativa_literatura.png`

**Qué muestra:** posicionamiento del TFG frente a resultados reportados en literatura (NDCG@10 aproximado).

**Referencias incluidas:** BPR-MF, Markov puro, FPMC, Item-KNN (ML tradicional); GRU4Rec, GETNext, STHGCN (Deep Learning SOTA).

**Nota metodológica:** la media se calcula **solo sobre Osaka + Petaling Jaya**. Istanbul se excluye porque su volumen de datos es ~4× menor que el de las otras ciudades (161K check-ins vs. 675K en Osaka y 460K en PJ). Esta diferencia produce resultados estructuralmente inferiores que no reflejan el comportamiento del sistema en condiciones normales. Además, el TFG usa NDCG@20 y protocolo `last_trail_user` (trail recommendation), mientras que la literatura usa NDCG@10 y leave-one-out (next-POI prediction) — la comparación es orientativa, no directa.

**Qué se lee de la figura:** los motores del TFG superan todos los baselines ML tradicionales. El mejor motor (Item-Item) iguala GRU4Rec y queda por debajo del SOTA de deep learning (GETNext, STHGCN). Esto es un resultado sólido para un TFG: superar los métodos clásicos y acercarse al estado del arte sin usar arquitecturas transformer ni GNNs.

---

### `fig_22_comparativa_hit.png`

**Qué muestra:** comparativa específica de Hit@K entre resultados del TFG y referencias de literatura.

**Nota:** media calculada sobre **Osaka + Petaling Jaya** (mismo criterio que fig_21 — Istanbul excluida por las mismas razones). La nota al pie de la figura explica explícitamente la exclusión y la diferencia de protocolo.

**Cómo interpretarla:** complementa fig_21 pero con la métrica Hit en lugar de nDCG. Hit mide si el sistema acertó aunque sea en una posición baja de la lista; nDCG penaliza los aciertos en posiciones bajas. Comparar ambas figuras permite ver si los motores aciertan pero en posiciones poco útiles (Hit alto, nDCG bajo) o si realmente ponen lo relevante arriba.

---

## 6) Dataset y comportamiento de usuarios

### `fig_16_distribucion_categorias.png`

**Qué muestra:** distribución por categoría amplia en Osaka.

**Codificación:** barras `POIs` = inventario disponible; barras `check-ins/10` = uso real histórico escalado.

**Cómo interpretarla:** compara la oferta (cuántos POIs hay de cada tipo) con la demanda (cuántos check-ins acumula cada tipo). Una categoría que tiene pocas barras de POIs pero muchos check-ins es una categoría "saturada" donde pocos lugares concentran mucho tráfico. Una con muchos POIs pero pocos check-ins tiene mucha oferta infrautilizada.

**Qué se lee de la figura:** Food domina el uso real (muchos check-ins) aunque no sea la categoría con más POIs en el inventario. Transport tiene muchos POIs por su función de paso, pero relativamente pocos check-ins registrados (la gente pasa pero no hace check-in). Esto explica por qué el sistema necesita tratar Food y Transport de forma especial.

---

### `fig_17_long_tail_usuarios.png`

**Qué muestra:** distribución de actividad de usuarios en dos paneles.

**Cómo interpretarla:** esta figura tiene dos partes que dicen lo mismo de dos formas distintas.

**Panel izquierdo (histograma log):**
El eje X es "cuántos check-ins tiene un usuario". El eje Y (en escala logarítmica) es "cuántos usuarios tienen esa cantidad". Lo que ves: la barra más alta está en la izquierda (usuarios con muy pocos check-ins) y cae rápidamente. Hay poquísimos usuarios con más de 500 check-ins.

En palabras simples: la mayoría de usuarios visitaron pocos sitios, y unos pocos "superfans" visitaron muchísimos.

**Panel derecho (Curva de Lorenz):**
Esta curva mide la desigualdad. Si fuera perfectamente igual, sería la línea diagonal punteada. Cuanto más se aleja hacia abajo de esa diagonal, más desigual es la distribución.

**Qué se lee de la figura:** la curva roja se aleja mucho de la diagonal. Esto significa que aproximadamente el **20% de los usuarios más activos generan el 80% de todos los check-ins del dataset** (efecto Pareto). Este fenómeno tiene implicaciones directas para los motores colaborativos: están muy bien entrenados para los usuarios activos, pero tienen poca información para los usuarios esporádicos (cold start).

---

### `fig_18_heatmap_temporal.png`

**Qué muestra:** patrón temporal hora × día de semana de check-ins en Osaka.

**Codificación:** eje X = hora del día (0–23); eje Y = día de la semana; color = número de check-ins.

**Cómo interpretarla:** es el "horario turístico" de Osaka. Las celdas más oscuras son las franjas de mayor actividad. Los picos habituales son en horas de comida (mediodía y noche) y en fin de semana. Las celdas más claras indican momentos de baja actividad (madrugada, días laborables de mañana).

**Qué se lee de la figura:** el turismo en Osaka sigue patrones temporales claros. Esta información es relevante para entender el contexto en que se generaron los datos y para futuros sistemas que incorporen contexto temporal en las recomendaciones.

---

### `fig_19_longitud_trails.png`

**Qué muestra:** histograma de longitud de trails (número de POIs por trail).

**Codificación:** barras = frecuencia; líneas verticales = media y mediana.

**Cómo interpretarla:** responde a "¿cuántos sitios visita un turista en una salida típica?". La mayoría de los trails tienen 2–5 POIs — las rutas turísticas reales son cortas. El sistema recomienda hasta 20 POIs (k=20) para dar flexibilidad, pero la longitud típica de una ruta real que el usuario haría está en ese rango corto.

**Qué se lee de la figura:** hay una cola larga hacia la derecha (algunos usuarios tienen trails muy largos), pero la mediana y la media están relativamente bajas. Esto valida el diseño del sistema: recomendar rutas de 5–10 POIs es razonable y coherente con el comportamiento real observado.

---

## 7) Figuras de dataset y esquema (`data/reports/figures/dataset/`)

Generadas por `scripts/generate_dataset_figures.py`. Carpeta independiente de las figuras de la memoria principal.

---

### `fig_er_diagram.png`

**Qué muestra:** diagrama entidad-relación del esquema PostgreSQL con las 4 tablas principales.

**Tablas:** `visits`, `pois`, `poi_categories`, `saved_routes`.

**Relaciones:** visits N:1 pois (via `venue_id → fsq_id`); pois 1:N poi_categories (via `fsq_id`); pois → saved_routes (conceptual via `city_qid`, línea discontinua).

**Codificación:** cabecera de tabla en color por tabla; PK en verde, FK en morado; filas alternas sombreadas.

**Cómo interpretarla:** es el mapa del terreno de datos. Antes de entender cómo funciona cualquier motor de recomendación, hay que entender de qué tablas se alimenta. Este diagrama es el punto de partida para el capítulo de diseño de base de datos. La relación central `visits ↔ pois` es la que genera toda la información de comportamiento de usuarios.

---

### `fig_etl_flow.png`

**Qué muestra:** pipeline ETL completo desde el CSV de Foursquare crudo hasta las tablas PostgreSQL.

**Flujo:** `Foursquare CSV → 01_clean_std → 02_extract_ids → 03_fetch_pois → 04_normalize_pois → 05_label_categories → 06_impute_pois → 07_diagnostics → 08_load_postgres → PostgreSQL DB`.

**Codificación:** cada caja es un script ETL; las etiquetas laterales son los artefactos que produce cada paso (std_clean.csv, venue_ids.txt, pois_raw.json, etc.).

**Cómo interpretarla:** de arriba hacia abajo; cada paso transforma los datos de una forma específica. El ETL no es trivial: el CSV bruto de Foursquare tiene datos ruidosos, IDs sin información, categorías en formato no normalizado. Los 8 pasos sistemáticamente van limpiando, enriqueciendo y normalizando hasta llegar a las tablas finales. Permite entender qué proceso hay detrás de los datos que usan los motores.

---

### `fig_bubble_dataset.png`

**Qué muestra:** las 3 ciudades como burbujas cuyo tamaño es proporcional a √(check-ins); dentro de cada burbuja, un donut con la distribución de categorías amplias.

**Codificación:** tamaño de burbuja = volumen de datos (raíz cuadrada de check-ins); secciones del donut = categorías; leyenda compartida en la parte inferior.

**Cómo interpretarla:** permite comparar volumen relativo y composición temática de cada ciudad de un solo vistazo. Osaka (~675K check-ins) es visiblemente mayor que Istanbul (~161K) y Petaling Jaya (~460K). La composición del donut muestra si cada ciudad tiene un perfil turístico distinto (más Food, más Culture, etc.).

**Qué se lee de la figura:** Osaka domina en volumen, lo que explica por qué es la ciudad de referencia del TFG y la que produce los mejores resultados de evaluación. Istanbul, con el donut más pequeño, tiene una distribución de categorías diferente, lo que también contribuye a sus resultados más bajos.

---

### `fig_heatmap_coverage.png`

**Qué muestra:** matriz ciudad × 6 estadísticas del dataset (usuarios únicos, POIs, check-ins, trails, media POIs/trail, sparsidad).

**Codificación:** coloreado por intensidad relativa dentro de cada columna (normalización 0–1 por métrica); cada celda muestra el valor real.

**Cómo interpretarla:** es la "ficha técnica" comparativa de las 3 ciudades en una sola tabla. Permite ver de un vistazo por qué Istanbul es más difícil (menos datos en todas las métricas, más sparsidad) que Osaka. La columna de sparsidad es especialmente relevante: una sparsidad alta significa que la matriz usuario-POI tiene muy pocos datos, lo que perjudica directamente a los motores colaborativos (ALS, Item-Item).

**Qué se lee de la figura:** explica visualmente las diferencias de rendimiento entre ciudades observadas en fig_12–fig_22. No es que los motores fallen en Istanbul por un bug — es que simplemente hay ~4× menos datos para aprender.

---

## 8) Análisis cold/warm de usuarios

### `fig_26_cold_warm_breakdown.png`

**Qué muestra:** comparativa Hit@20 y nDCG@20 para usuarios cold (<5 visitas en train) vs. warm (≥5).

**Codificación:** layout 2 filas (métricas: Hit@20 y nDCG@20) × 3 columnas (ciudades); barras agrupadas por motor (hybrid, rrf, item, markov, als, embed); azul = cold, rojo = warm.

**Cómo interpretarla:**
- "Cold" = usuario con menos de 5 visitas en el entrenamiento (casi desconocido para el sistema).
- "Warm" = usuario con 5 o más visitas (el sistema lo conoce bien).

**Qué se lee de la figura:** las barras frías (cold) son más altas que las cálidas (warm) en la mayoría de motores. Parece raro — ¿los desconocidos son más fáciles de recomendar?

La razón es estadística: los usuarios cold tienen muy pocos trails reales en el test, así que con recomendar cualquier cosa popular ya hay más probabilidad de acertar por pura aleatoriedad. No es que el sistema sea mejor con ellos — es que el denominador del cálculo es pequeño.

Lo importante para el TFG: **ALS y Embed caen a casi cero en Istanbul para usuarios cold** — eso indica que sin datos suficientes, los motores colaborativos fallan completamente (necesitan al menos algunos vectores aprendidos para funcionar). **Markov e Hybrid son los más robustos ante este problema**: Markov porque puede recomendar transiciones frecuentes globales aunque no conozca al usuario; Hybrid porque diversifica entre motores y no depende de uno solo.

---

## Nota de interpretación técnica

- Algunas figuras con basemap (`fig_02`, `fig_03`, `fig_04`, `fig_05`, `fig_23`, `fig_24`, `fig_25`) dependen de acceso a proveedores de tiles web (CartoDB, OSM). Si hay bloqueo de red/firewall en el entorno Python, se puede ver fondo neutro en vez de mapa base. Esto **no invalida los datos de la figura**, solo afecta la capa visual de contexto cartográfico.
- Las figuras `fig_23` y `fig_25` muestran solo **Osaka** (mayor volumen de datos, resultados más representativos). `fig_24` mantiene las 3 ciudades en subplots.
- Los resultados de evaluación recogidos en fig_12–fig_22 y fig_26 corresponden al benchmark con protocolo `last_trail_user --fair`, seed=42, max_users=300, k=20.
- Las figuras fig_14, fig_14b, fig_21 y fig_22 usan la **media Osaka + Petaling Jaya** (Istanbul excluida por volumen ~4× inferior — ver nota detallada en la sección de fig_21).
