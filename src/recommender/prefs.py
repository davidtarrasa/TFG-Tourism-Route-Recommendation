"""
User preferences parsing for the CLI.

Goal: keep the CLI simple (one --prefs string) while mapping it to concrete
filters/boosts in the recommender.

Supported tokens (case-insensitive, comma-separated):
- Categories: any token not recognized as a keyword is treated as a category preference.
- free: prefer/require POIs with is_free = True
- paid: prefer/require POIs with is_free = False
- cheap|budget: max_price_tier = 1
- mid: max_price_tier = 2
- expensive: max_price_tier = 3
- price:0..4 or max_price:0..4: explicit max_price_tier

Examples:
  --prefs "museum,park,free,cheap"
  --prefs "Japanese Restaurant,paid,price:2"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Prefs:
    categories: List[str]
    free_only: Optional[bool] = None  # True=only free, False=only paid, None=no filter
    max_price_tier: Optional[int] = None


def _to_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def parse_prefs(prefs: Optional[str]) -> Prefs:
    if not prefs:
        return Prefs(categories=[])

    tokens = [t.strip() for t in prefs.split(",") if t.strip()]
    categories: List[str] = []
    free_only: Optional[bool] = None
    max_price_tier: Optional[int] = None

    for raw in tokens:
        t = raw.strip()
        tl = t.lower()

        if tl in ("free", "gratis"):
            free_only = True
            continue
        if tl in ("paid", "pago"):
            free_only = False
            continue

        if tl in ("cheap", "budget", "barato"):
            max_price_tier = 1
            continue
        if tl in ("mid", "medio"):
            max_price_tier = 2
            continue
        if tl in ("expensive", "caro"):
            max_price_tier = 3
            continue

        if tl.startswith(("price:", "max_price:", "max-price:", "price_tier:", "price-tier:")):
            val = _to_int(t.split(":", 1)[1].strip())
            if val is not None:
                max_price_tier = val
                continue

        # Default: treat as category preference.
        categories.append(t)

    return Prefs(categories=categories, free_only=free_only, max_price_tier=max_price_tier)


def normalize_categories(categories: Sequence[str]) -> List[str]:
    """Basic normalization for matching: strip, drop empties, keep original casing."""
    out: List[str] = []
    seen = set()
    for c in categories:
        cc = (c or "").strip()
        if not cc:
            continue
        key = cc.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cc)
    return out


__all__ = ["Prefs", "parse_prefs", "normalize_categories"]

