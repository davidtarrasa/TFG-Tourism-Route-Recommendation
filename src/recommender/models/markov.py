"""
Modelos secuenciales:
- Matriz de transiciones POI→POI y categoría→categoría.
- Dado current_poi o current_category, sugerir siguientes con mayor probabilidad.
- Empates: se resolverán en el scorer (distancia, rating, etc.).
"""

from typing import Dict, List, Tuple, Optional


def next_poi(
    current_poi: str,
    transitions_poi: Dict[str, Dict[str, float]],
    topn: int = 20,
) -> List[Tuple[str, float]]:
    """Devuelve candidatos ordenados por probabilidad desde un POI actual (orden 1)."""
    probs = transitions_poi.get(current_poi, {})
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:topn]


def next_poi_order2(
    prev_poi: Optional[str],
    current_poi: str,
    transitions_order2: Dict[Tuple[str, str], Dict[str, float]],
    transitions_order1: Dict[str, Dict[str, float]],
    topn: int = 20,
    backoff: float = 0.3,
) -> List[Tuple[str, float]]:
    """
    Markov de orden 2 con backoff a orden 1.

    score = (1-backoff)*P2(next|prev,current) + backoff*P1(next|current)
    """
    p1 = transitions_order1.get(current_poi, {})
    p2 = transitions_order2.get((prev_poi, current_poi), {}) if prev_poi else {}
    if not p1 and not p2:
        return []

    scores: Dict[str, float] = {}
    keys = set(p1) | set(p2)
    for k in keys:
        scores[k] = (1.0 - backoff) * p2.get(k, 0.0) + backoff * p1.get(k, 0.0)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]


def next_category(current_category: str, transitions_cat: Dict[str, Dict[str, float]], topn: int = 20) -> List[Tuple[str, float]]:
    """Devuelve candidatos de categorías ordenados por probabilidad."""
    probs = transitions_cat.get(current_category, {})
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:topn]


__all__ = ["next_poi", "next_poi_order2", "next_category"]
