"""
Modelos secuenciales:
- Matriz de transiciones POI→POI y categoría→categoría.
- Dado current_poi o current_category, sugerir siguientes con mayor probabilidad.
- Empates: se resolverán en el scorer (distancia, rating, etc.).
"""

from typing import Dict, List, Tuple


def next_poi(current_poi: str, transitions_poi: Dict[str, Dict[str, float]], topn: int = 20) -> List[Tuple[str, float]]:
    """Devuelve candidatos ordenados por probabilidad desde un POI actual."""
    probs = transitions_poi.get(current_poi, {})
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:topn]


def next_category(current_category: str, transitions_cat: Dict[str, Dict[str, float]], topn: int = 20) -> List[Tuple[str, float]]:
    """Devuelve candidatos de categorías ordenados por probabilidad."""
    probs = transitions_cat.get(current_category, {})
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:topn]


__all__ = ["next_poi", "next_category"]
