"""
Category intent mapping utilities.

Goal:
- Map noisy POI categories into a compact set of user-facing intents.
- Support a fallback "inconclusive" bucket.
- Keep inference lightweight (rules first), with optional semantic model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json
import os
import re
import unicodedata

INTENTS: Tuple[str, ...] = (
    "food",
    "culture",
    "nature",
    "nightlife",
    "shopping",
    "service",
    "health",
    "entertainment",
    "transport",
    "relaxation",
    "family",
    "sports",
)

INCONCLUSIVE = "inconclusive"

# User tokens -> compact intent labels.
USER_INTENT_ALIASES: Dict[str, str] = {
    "food": "food",
    "restaurant": "food",
    "restaurants": "food",
    "cafe": "food",
    "café": "food",
    "cafes": "food",
    "coffee": "food",
    "Cha Chaan Teng": "food",
    "Kafenio": "food",
    "bar": "nightlife",
    "bars": "nightlife",
    "nightlife": "nightlife",
    "club": "nightlife",
    "casino": "nightlife",
    "brewery": "nightlife",
    "shopping": "shopping",
    "shop": "shopping",
    "store": "shopping",
    "retail": "shopping",
    "mall": "shopping",
    "grocery": "shopping",
    "studio": "shopping",
    "winery": "shopping",
    "dispensary": "shopping",
    "service": "service",
    "motorcycle": "service",
    "car": "service",
    "wash": "service",
    "amenity": "service",
    "daycare": "service",
    "tattoo": "service",
    "salon": "service",
    "storage": "service",
    "doctor": "health",
    "hospital": "health",
    "pharmacy": "health",
    "dentist": "health",
    "emergency room": "health",
    "optometrist": "health",
    "medical": "health",
    "medicine": "health",
    "clinic": "health",
    "dermatologist": "health",
    "chiropractor": "health",
    "surgeon": "health",
    "culture": "culture",
    "museum": "culture",
    "museums": "culture",
    "art": "culture",
    "gallery": "culture",
    "theater": "culture",
    "university": "culture",
    "college": "culture",
    "school": "culture",
    "pier": "culture",
    "port": "culture",
    "bridge": "culture",
    "landmark": "culture",
    "place": "culture",
    "site": "culture",
    "canal": "culture",
    "palace": "culture",
    "hall": "culture",
    "town": "culture",
    "city": "culture",
    "fountain": "culture",
    "harbor": "culture",
    "structure": "culture",
    "government": "culture",
    "neighborhood": "culture",
    "mosque": "culture",
    "intersection": "culture",
    "church": "culture",
    "capitol": "culture",
    "base": "culture",
    "cemetery": "culture",
    "embassy": "culture",
    "building": "culture",
    "center": "culture",
    "courthouse": "culture",
    "bank": "culture",
    "village": "culture",
    "tunnel": "culture",
    "synagogue": "culture",
    "crossing": "culture",
    "state": "culture",
    "county": "culture",
    "park": "culture",
    "nature": "nature",
    "garden": "nature",
    "tree": "nature",
    "spring": "nature",
    "animal": "nature",
    "hill": "nature",
    "field": "nature",
    "farm": "nature",
    "vineyard": "nature",
    "reservoir": "nature",
    "zoo": "family",
    "family": "family",
    "kid": "family",
    "kids": "family",
    "fair": "family",
    "event": "family",
    "launch": "family",
    "sport": "sports",
    "sports": "sports",
    "stadium": "sports",
    "gym": "sports",
    "golf": "sports",
    "surf": "sports",
    "climbing": "sports",
    "fishing": "sports",
    "target": "sports",
    "batting": "sports",
    "tennis": "sports",
    "skating": "sports",
    "play": "sports",
    "track": "sports",
    "race": "sports",
    "dive": "sports",
    "court": "sports",
    "pitch": "sports",
    "spa": "relaxation",
    "onsen": "relaxation",
    "relax": "relaxation",
    "movie": "entertainment",
    "cinema": "entertainment",
    "arcade": "entertainment",
    "laser tag": "entertainment",
    "circus": "entertainment",
    "gun range": "entertainment",
    "plane": "entertainment",
    "prison": "entertainment",
    "lighthouse": "entertainment",
    "planetarium": "entertainment",
    "house": "entertainment",
    "research": "entertainment",
    "island": "entertainment",
    "parlor": "entertainment",
    "sauna": "entertainment",
    "escape room": "entertainment",
    "dealership": "entertainment",
    "transport": "transport",
    "station": "transport",
    "metro": "transport",
    "airport": "transport",
    "taxi": "transport",
}

# Intent -> keyword hints (lowercase)
INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "food": (
        "restaurant",
        "food",
        "cafe",
        "café",
        "coffee",
        "breakfast",
        "chicken",
        "bistro",
        "spot",
        "creperie",
        "joint",
        "tea",
        "bakery",
        "sushi",
        "ramen",
        "bbq",
        "noodle",
        "pizza",
        "burger",
        "ice cream",
        "juice",
        "butcher",
        "Cha Chaan Teng",
        "Kafenio"
    ),
    "culture": (
        "museum",
        "gallery",
        "temple",
        "shrine",
        "castle",
        "historic",
        "history",
        "theater",
        "concert hall",
        "art",
        "monument",
        "public art",
        "opera",
        "library",
        "university",
        "college",
        "school",
        "pier",
        "port",
        "bridge",
        "landmark",
        "place",
        "site",
        "canal",
        "palace",
        "hall",
        "town",
        "city",
        "fountain",
        "harbor",
        "structure",
        "government",
        "neighborhood",
        "mosque",
        "intersection",
        "church",
        "capitol",
        "base",
        "cemetery",
        "embassy",
        "center",
        "courthouse",
        "bank",
        "village",
        "tunnel",
        "synagogue",
        "crossing",
        "state",
        "county",
        "park",
        "building"
    ),
    "nature": (
        "park",
        "garden",
        "tree",
        "spring",
        "animal",
        "hill",
        "field",
        "farm",
        "vineyard",
        "reservoir",
        "botanical",
        "forest",
        "beach",
        "mountain",
        "lake",
        "river",
        "trail",
    ),
    "nightlife": (
        "night",
        "bar",
        "pub",
        "club",
        "speakeasy",
        "cocktail",
        "karaoke",
        "casino",
        "brewery",
        "roof",
    ),
    "shopping": (
        "shop",
        "store",
        "mall",
        "grocery",
        "market",
        "boutique",
        "department store",
        "thrift",
        "vintage",
        "studio",
        "winery",
        "dispensary",
        "retail"
    ),
    "service": (
        "service",
        "motorcycle",
        "car",
        "wash",
        "amenity",
        "daycare",
        "tattoo",
        "salon",
        "barber",
        "spa service",
        "storage"
    ),
    "health": (
        "doctor",
        "hospital",
        "pharmacy",
        "dentist",
        "emergency room",
        "optometrist",
        "medical",
        "medicine",
        "clinic",
        "dermatologist",
        "chiropractor",
        "surgeon",
    ),
    "entertainment": (
        "arcade",
        "movie",
        "cinema",
        "music venue",
        "theme park",
        "amusement",
        "bowling",
        "games",
        "laser tag",
        "circus",
        "gun range",
        "plane",
        "prison",
        "lighthouse",
        "planetarium",
        "house",
        "research",
        "island",
        "parlor",
        "sauna",
        "escape room",
        "dealership",
    ),
    "transport": (
        "station",
        "metro",
        "rail",
        "bus",
        "airport",
        "terminal",
        "platform",
        "tram",
        "ferry",
        "taxi",
    ),
    "relaxation": (
        "spa",
        "onsen",
        "bath",
        "yoga",
        "wellness",
        "massage",
        "resort",
    ),
    "family": (
        "zoo",
        "aquarium",
        "playground",
        "toy",
        "kids",
        "family",
        "fair",
        "event",
        "launch",
    ),
    "sports": (
        "stadium",
        "arena",
        "sport",
        "gym",
        "golf",
        "surf",
        "climbing",
        "fishing",
        "target",
        "batting",
        "tennis",
        "skating",
        "play",
        "track",
        "race",
        "spot",
        "dive",
        "court",
        "pitch",
        "soccer",
        "baseball",
        "basketball",
        "pool",
        "fitness",
    ),
}

_DEFAULT_OVERRIDES_PATH = os.path.join("configs", "category_intent_overrides.json")


def _norm_text(s: str) -> str:
    t = (s or "").strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"\s+", " ", t)
    return t


@lru_cache(maxsize=1)
def _load_overrides(path: str = _DEFAULT_OVERRIDES_PATH) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        out: Dict[str, str] = {}
        for k, v in data.items():
            kk = _norm_text(str(k))
            vv = _norm_text(str(v))
            if vv in INTENTS or vv == INCONCLUSIVE:
                out[kk] = vv
        return out
    except Exception:
        return {}


def _keyword_intent(category_name: str) -> Tuple[str, float]:
    t = _norm_text(category_name)
    if not t:
        return INCONCLUSIVE, 0.0

    def _has_kw(text: str, kw: str) -> bool:
        # Match whole terms (or full phrases), avoiding accidental partial hits:
        # e.g. "port" should not match "airport" or "sports".
        pattern = r"(?<!\\w)" + re.escape(kw) + r"(?!\\w)"
        return re.search(pattern, text) is not None

    best_intent = INCONCLUSIVE
    best_score = 0.0
    for intent, kws in INTENT_KEYWORDS.items():
        score = 0.0
        for kw in kws:
            if _has_kw(t, kw):
                score += 1.0
        if score > best_score:
            best_score = score
            best_intent = intent
    if best_score <= 0:
        return INCONCLUSIVE, 0.0
    # Confidence proxy in [0,1], enough for downstream weighting.
    conf = min(1.0, 0.35 + 0.15 * best_score)
    return best_intent, conf


@lru_cache(maxsize=1)
def _semantic_backend() -> Optional[object]:
    """
    Optional semantic backend:
    - sentence-transformers if installed (preferred)
    - otherwise None
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


def _semantic_intent(category_name: str) -> Tuple[str, float]:
    model = _semantic_backend()
    if model is None:
        return INCONCLUSIVE, 0.0
    try:
        import numpy as np

        labels = list(INTENTS) + [INCONCLUSIVE]
        # Simple prompt-style label texts.
        label_texts = [f"tourism category intent: {x}" for x in labels]
        vec_cat = model.encode([f"poi category: {category_name}"], normalize_embeddings=True)[0]
        vec_labels = model.encode(label_texts, normalize_embeddings=True)
        sims = np.dot(vec_labels, vec_cat)
        idx = int(np.argmax(sims))
        conf = float(max(0.0, sims[idx]))
        return labels[idx], conf
    except Exception:
        return INCONCLUSIVE, 0.0


@lru_cache(maxsize=20000)
def classify_category_intent(
    category_name: str,
    use_semantic: bool = True,
    semantic_threshold: float = 0.42,
) -> Tuple[str, float, str]:
    """
    Return: (intent, confidence, source)
    source in {"override", "keyword", "semantic", "inconclusive"}.
    """
    t = _norm_text(category_name)
    if not t:
        return INCONCLUSIVE, 0.0, "inconclusive"

    overrides = _load_overrides()
    if t in overrides:
        return overrides[t], 1.0, "override"

    intent_kw, conf_kw = _keyword_intent(t)
    if intent_kw != INCONCLUSIVE and conf_kw >= 0.45:
        return intent_kw, conf_kw, "keyword"

    if use_semantic:
        intent_sem, conf_sem = _semantic_intent(t)
        if intent_sem != INCONCLUSIVE and conf_sem >= semantic_threshold:
            return intent_sem, conf_sem, "semantic"

    if intent_kw != INCONCLUSIVE:
        return intent_kw, conf_kw, "keyword"
    return INCONCLUSIVE, 0.0, "inconclusive"


def infer_user_intents(categories: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in categories:
        token = _norm_text(str(raw))
        if not token:
            continue
        if token in USER_INTENT_ALIASES:
            mapped = USER_INTENT_ALIASES[token]
            if mapped not in seen:
                seen.add(mapped)
                out.append(mapped)
            continue
        intent, _, _ = classify_category_intent(token, use_semantic=True)
        if intent != INCONCLUSIVE and intent not in seen:
            seen.add(intent)
            out.append(intent)
    return out


def classify_category_set(category_names: Iterable[str]) -> Set[str]:
    intents: Set[str] = set()
    for c in category_names:
        i, _, _ = classify_category_intent(str(c), use_semantic=True)
        if i:
            intents.add(i)
    return intents


__all__ = [
    "INTENTS",
    "INCONCLUSIVE",
    "classify_category_intent",
    "classify_category_set",
    "infer_user_intents",
]
