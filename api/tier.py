"""
api/tier.py

Tier gating for Ethos/Verum API.

Two tiers:
  free — score endpoint returns top 5 values only, no text excerpts,
         no certification, no export, no dataset compilation.
  pro  — full 15 values, text excerpts, certification, export, dataset.

Gating is API-key based:
  - No key or missing key = free tier.
  - Valid pro key = pro tier.

Pro keys are stored in data/api_keys.json (created on first run).
Use cli tools or direct file editing to manage keys.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException

_log = logging.getLogger(__name__)

_KEYS_PATH = Path(__file__).resolve().parent.parent / "data" / "api_keys.json"

# Free tier: only these 5 values are returned in score responses
FREE_TIER_VALUES = frozenset({
    "integrity",
    "courage",
    "compassion",
    "resilience",
    "responsibility",
})

FREE_TIER_MAX_SIGNALS = 5


def _load_pro_keys() -> set[str]:
    """Load pro API keys from data/api_keys.json."""
    if not _KEYS_PATH.exists():
        _KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _KEYS_PATH.write_text(json.dumps({"pro_keys": []}, indent=2))
        return set()
    try:
        data = json.loads(_KEYS_PATH.read_text(encoding="utf-8"))
        return set(data.get("pro_keys", []))
    except Exception as exc:
        _log.warning("Failed to load API keys: %s", exc)
        return set()


def resolve_tier(x_api_key: Optional[str] = Header(None)) -> str:
    """
    FastAPI dependency that resolves the caller's tier from the X-Api-Key header.
    Returns "pro" or "free".
    """
    if not x_api_key:
        return "free"
    pro_keys = _load_pro_keys()
    if x_api_key in pro_keys:
        return "pro"
    return "free"


def require_pro(tier: str) -> None:
    """Raise 403 if the caller is not on the pro tier."""
    if tier != "pro":
        raise HTTPException(
            status_code=403,
            detail="This endpoint requires a Pro subscription. Visit https://trust-forged.com for details.",
        )


def filter_signals_for_tier(signals: list[dict], tier: str) -> list[dict]:
    """
    Filter score response signals based on tier.

    Free tier:
      - Only signals for the 5 free-tier values
      - text_excerpt stripped (replaced with indicator)
      - embedding_score stripped
      - Capped at FREE_TIER_MAX_SIGNALS

    Pro tier:
      - All signals returned unmodified
    """
    if tier == "pro":
        return signals

    filtered = []
    for s in signals:
        if s.get("value_name") not in FREE_TIER_VALUES:
            continue
        # Strip detailed fields from free tier
        s_copy = dict(s)
        s_copy["text_excerpt"] = "[Pro subscription required for text excerpts]"
        s_copy.pop("embedding_score", None)
        filtered.append(s_copy)
        if len(filtered) >= FREE_TIER_MAX_SIGNALS:
            break

    return filtered
