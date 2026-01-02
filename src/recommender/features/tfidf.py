"""
Construcción de matriz TF-IDF de categorías por POI.
- Entrada: DataFrame con columnas fsq_id, category_name.
- Salida: (matriz TF-IDF sparse, lista de fsq_id en el mismo orden, vectorizer)
"""

from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(poi_categories_df: pd.DataFrame) -> Tuple:
    """Agrupa categorías por POI y construye TF-IDF."""
    if poi_categories_df.empty:
        raise ValueError("poi_categories_df está vacío.")

    # Agrupar categorías por POI y concatenar nombres
    grouped = poi_categories_df.groupby("fsq_id")["category_name"].apply(list).reset_index()
    grouped["text"] = grouped["category_name"].apply(lambda cats: " ".join(cats))

    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(grouped["text"])

    fsq_ids: List[str] = grouped["fsq_id"].tolist()
    return tfidf_matrix, fsq_ids, vectorizer


__all__ = ["build_tfidf"]
