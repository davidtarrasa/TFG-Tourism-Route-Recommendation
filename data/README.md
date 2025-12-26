Data layout
===========

- processed/ : datos listos para cargar (std_clean.csv, pois_enriched_*.json, category_price_labels.json, etc.).
- raw/ : coloca aquí los datos brutos cuando los tengas (std_2018.csv, POIs del profesor, respuestas de APIs, etc.).

Qué falta (TODO)
----------------
- Añadir enlaces de descarga a std_2018.csv y los JSON/CSVs del profesor. TODO: subirlos a un bucket o release y documentar aquí las URLs.
- Si se regeneran los procesados, actualizar este README con los comandos usados.

Carga en Postgres (resumen)
---------------------------
1) Asegura que processed/ contiene:
   - data/processed/std_clean.csv
   - data/processed/pois_enriched_*.json
   - data/processed/category_price_labels.json
2) Levanta Docker y carga:
   - docker compose up -d db pgadmin
   - python src/etl/08_load_postgres.py --dsn postgresql://tfg:tfgpass@localhost:55432/tfg_routes

Nota: los datos brutos en raw/ suelen ser pesados o con restricciones, por eso no se versionan. Los procesados sí están incluidos para que el arranque sea inmediato.
