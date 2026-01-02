"""
Entrena un modelo Word2Vec (Skip-gram) sobre secuencias de visits y lo guarda en cache.

Uso:
  python -m src.recommender.train_embeddings --city-qid Q35765 --visits-limit 200000 --out src/recommender/cache/word2vec_osaka.joblib

Requiere: gensim, joblib.
"""

import argparse
import os

import joblib

from .features.load_data import load_all
from .features.word2vec import sequences_from_visits
from .models.embeddings import train_embeddings


def main():
    parser = argparse.ArgumentParser(description="Entrena embeddings Word2Vec sobre rutas (visits)")
    parser.add_argument("--dsn", help="DSN de Postgres (si no, usa POSTGRES_DSN o por defecto)")
    parser.add_argument("--city", help="Filtro opcional sobre pois.city")
    parser.add_argument("--city-qid", dest="city_qid", help="Filtro opcional sobre visits.venue_city (QID)")
    parser.add_argument("--visits-limit", dest="visits_limit", type=int, help="Limitar visits (para acelerar)")
    parser.add_argument("--out", default="src/recommender/cache/word2vec.joblib", help="Ruta de salida del modelo")
    parser.add_argument("--vector-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    visits_df, _, _ = load_all(dsn=args.dsn, city=args.city, city_qid=args.city_qid, visits_limit=args.visits_limit)
    seqs = sequences_from_visits(visits_df)
    if not seqs:
        raise SystemExit("No hay secuencias para entrenar embeddings (visits vacío o filtro demasiado restrictivo).")

    model = train_embeddings(
        seqs,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Modelo Word2Vec guardado en {args.out}")


if __name__ == "__main__":
    main()
