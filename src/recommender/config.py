"""Config loader for the recommender.

We use TOML so we don't need extra dependencies (Python 3.11+ ships tomllib).
"""

from __future__ import annotations

from dataclasses import dataclass
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


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config from TOML, or return defaults if missing."""
    path = path or DEFAULT_CONFIG_PATH

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
    }

    if not os.path.exists(path):
        return defaults

    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _deep_update(defaults, data)


__all__ = ["DEFAULT_CONFIG_PATH", "load_config"]

