"""Config loader for the recommender.

We use TOML so we don't need extra dependencies (Python 3.11+ ships tomllib).
Supports per-city config files:
  - configs/recommender.toml (global fallback)
  - configs/recommender_<city_qid>.toml (city-specific override, e.g. recommender_q35765.toml)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import os


DEFAULT_CONFIG_PATH = os.path.join("configs", "recommender.toml")


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def city_config_path(city_qid: str, base_path: Optional[str] = None) -> str:
    """Build city-specific config path from QID."""
    base = base_path or DEFAULT_CONFIG_PATH
    city = str(city_qid).strip().lower()
    directory = os.path.dirname(base) or "."
    return os.path.join(directory, f"recommender_{city}.toml")


def resolve_config_path(path: Optional[str] = None, city_qid: Optional[str] = None) -> str:
    """
    Resolve config path.

    Rules:
    - If explicit `path` is provided and is not the default path, use it as-is.
    - Otherwise, if `city_qid` is provided and a city config exists, use it.
    - Fallback to `path` or default.
    """
    target = path or DEFAULT_CONFIG_PATH
    if path and os.path.normpath(path) != os.path.normpath(DEFAULT_CONFIG_PATH):
        return target
    if city_qid:
        cpath = city_config_path(city_qid, base_path=DEFAULT_CONFIG_PATH)
        if os.path.exists(cpath):
            return cpath
    return target


def load_config(path: Optional[str] = None, city_qid: Optional[str] = None) -> Dict[str, Any]:
    """Load config from TOML, or return defaults if missing."""
    path = resolve_config_path(path=path, city_qid=city_qid)

    defaults: Dict[str, Any] = {
        "embeddings": {"vector_size": 128, "window": 15, "min_count": 2, "workers": 4, "topn_score": 1000},
        "als": {"factors": 128, "regularization": 0.01, "iterations": 30, "alpha": 40.0, "topn_score": 500},
        "hybrid": {
            "user_current": [0.1, 0.15, 0.15, 0.05, 0.55],
            "user_only": [0.1, 0.2, 0.1, 0.05, 0.55],
            "current_only": [0.1, 0.15, 0.35, 0.05, 0.35],
            "embed_or_als": [0.15, 0.15, 0.1, 0.15, 0.45],
            "cold_start": [0.6, 0.2, 0.2, 0.0, 0.0],
        },
        "filters": {"exclude_categories": ["Intersection", "State", "Home (private)"]},
        "prefs": {
            "category_boost": 0.2,
            "intent_boost": 0.3,
            "inconclusive_penalty": 0.8,
            "strict_min_confidence": 0.35,
        },
        "category_intents": {
            "enabled": True,
            "use_semantic": True,
            "semantic_threshold": 0.42,
        },
    }

    if not os.path.exists(path):
        return defaults

    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _deep_update(defaults, data)


__all__ = ["DEFAULT_CONFIG_PATH", "city_config_path", "resolve_config_path", "load_config"]
