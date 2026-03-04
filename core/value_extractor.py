"""
core/value_extractor.py

Value signal extraction — keyword vocabulary match against 15 named values.

Architecture:
  process_figure(session_id)     — public entry point for the historical pipeline
  extract_value_signals(...)     — keyword vocabulary match
  compute_resistance(...)        — imported from core.resistance

Value Vocabulary (Phase 1):
  15 named values with keyword triggers (case-insensitive substring match).
  Phase 2 will add embedding-based clustering.

Constitutional invariants:
  No LLM call anywhere in this module.
  All entry points wrapped in try/except (fail-open).
"""

from __future__ import annotations

import time
from typing import Dict, List

from core.config import get_config
from core.document_store import get_document_store
from core.resistance import compute_resistance
from core.value_store import get_value_store

# ---------------------------------------------------------------------------
# Value vocabulary
# ---------------------------------------------------------------------------

VALUE_VOCAB: Dict[str, List[str]] = {
    "integrity": [
        "honest", "honesty", "truth", "truthful", "genuine", "sincere",
        "transparent", "real with", "can't lie", "won't lie", "tell you the truth",
    ],
    "courage": [
        "afraid", "scared", "brave", "bravery", "courage", "courageous",
        "risk", "risking", "hard to say", "nervous about", "terrified",
        "fear", "facing my fear",
    ],
    "compassion": [
        "care about", "worry about", "worrying about", "sad for", "feel for",
        "sorry for", "heart goes out", "feeling for", "concerned about them",
    ],
    "commitment": [
        "promise", "promised", "commit", "committed", "dedicated", "dedication",
        "always will", "I will", "won't give up", "I won't stop", "I'll be there",
    ],
    "patience": [
        "patient", "patience", "waiting", "wait it out", "take time",
        "slow down", "in time", "eventually", "let it unfold",
    ],
    "responsibility": [
        "my fault", "my responsibility", "responsible for", "should have",
        "accountable", "I owe", "I let", "I need to fix", "on me",
    ],
    "fairness": [
        "fair", "fairness", "equal", "equality", "just", "justice",
        "deserves", "unfair", "unjust", "not right",
    ],
    "gratitude": [
        "grateful", "gratitude", "thankful", "appreciate", "appreciation",
        "thank you", "lucky to have", "so glad", "means a lot",
    ],
    "curiosity": [
        "wondering", "wonder", "curious", "curiosity", "want to know",
        "interested in", "explore", "fascinated", "trying to understand",
    ],
    "resilience": [
        "keep going", "keep trying", "bounce back", "despite", "even though",
        "still going", "won't quit", "push through", "get through this",
    ],
    "love": [
        "love", "I love", "care deeply", "cherish", "miss", "missing",
        "mean everything", "everything to me", "means the world",
    ],
    "growth": [
        "better at", "getting better", "improve", "improving", "learning",
        "growing", "want to be", "trying to become", "working on myself",
    ],
    "independence": [
        "on my own", "by myself", "my choice", "my decision", "self-reliant",
        "don't need anyone", "figure it out myself",
    ],
    "loyalty": [
        "always there", "stand by", "standing by", "loyal", "loyalty",
        "won't leave", "I'll be there", "through thick and thin",
    ],
    "humility": [
        "I was wrong", "my mistake", "I made a mistake", "I learned",
        "I missed", "I didn't know", "I was mistaken", "I need to admit",
    ],
}

_MIN_KEYWORD_LEN = 3
_MAX_PASSAGES_PER_RUN = 200


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_figure(session_id: str) -> int:
    """
    Run value extraction for a historical figure session.
    Returns the number of observations recorded.
    Never raises.
    """
    try:
        return _run_extraction(session_id)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Core watermark-gated extraction loop
# ---------------------------------------------------------------------------

def _run_extraction(session_id: str) -> int:
    cfg = get_config()
    doc_store = get_document_store()
    val_store = get_value_store()

    watermark = doc_store.get_watermark(session_id)
    passages = doc_store.get_passages_since(session_id, watermark)
    if not passages:
        return 0

    recorded = 0
    now = time.time()
    latest_ts = watermark

    for row in passages[:_MAX_PASSAGES_PER_RUN]:
        text = str(row["text"])
        significance = float(row["significance"])
        doc_type = str(row["doc_type"])
        ts = float(row["ts"])
        record_id = str(row["id"])

        if significance < cfg.min_significance_threshold or not text:
            latest_ts = max(latest_ts, ts)
            continue

        signals = extract_value_signals(text, record_id, significance)
        for sig in signals:
            resistance = compute_resistance(text, significance, doc_type)
            if resistance < cfg.min_resistance_threshold:
                continue

            val_store.record_observation(
                session_id=session_id,
                turn_id=record_id,
                record_id=record_id,
                ts=ts,
                value_name=sig["value_name"],
                text_excerpt=sig["text_excerpt"],
                significance=significance,
                resistance=resistance,
            )
            val_store.upsert_registry(
                session_id=session_id,
                value_name=sig["value_name"],
                significance=significance,
                resistance=resistance,
                ts=ts,
            )
            # Cross-figure aggregate (session_id='')
            val_store.upsert_registry(
                session_id="",
                value_name=sig["value_name"],
                significance=significance,
                resistance=resistance,
                ts=ts,
            )
            recorded += 1

        latest_ts = max(latest_ts, ts)

    doc_store.set_watermark(session_id, latest_ts if latest_ts > watermark else now)
    return recorded


# ---------------------------------------------------------------------------
# Vocabulary-based signal extraction
# ---------------------------------------------------------------------------

def extract_value_signals(
    text: str,
    record_id: str,
    significance: float,
) -> List[Dict]:
    """
    Keyword-scan text against VALUE_VOCAB (case-insensitive substring).
    Returns a list of {value_name, text_excerpt, significance} dicts.
    """
    if not text:
        return []

    text_lower = text.lower()
    results = []
    seen_values: set = set()

    for value_name, keywords in VALUE_VOCAB.items():
        if value_name in seen_values:
            continue
        for kw in keywords:
            if len(kw) < _MIN_KEYWORD_LEN:
                continue
            if kw.lower() in text_lower:
                excerpt = _extract_excerpt(text, kw, max_len=150)
                results.append({
                    "value_name": value_name,
                    "text_excerpt": excerpt,
                    "significance": significance,
                })
                seen_values.add(value_name)
                break

    return results


def _extract_excerpt(text: str, keyword: str, max_len: int = 150) -> str:
    idx = text.lower().find(keyword.lower())
    if idx < 0:
        return text[:max_len]
    start = max(0, idx - 40)
    end = min(len(text), idx + len(keyword) + 60)
    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + "..."
    return excerpt[:max_len]
