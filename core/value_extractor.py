"""
core/value_extractor.py

Value signal extraction — keyword vocabulary match against 15 named values.

Architecture:
  process_figure(session_id)     — public entry point for the historical pipeline
  extract_value_signals(...)     — keyword vocabulary match
  compute_resistance(...)        — imported from core.resistance

Value Vocabulary (Phase 1):
  15 named values, ~25-35 keyword triggers each (case-insensitive substring match).
  Each value has: direct assertions, historical/formal diction, action-phrase patterns,
  and failure markers (for P0/APY signal).
  Phase 2 will add embedding-based clustering alongside this vocabulary.

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
        # Direct assertions
        "honest", "honesty", "truth", "truthful", "genuine", "sincere",
        "transparent", "real with", "can't lie", "won't lie", "tell you the truth",
        # Historical / formal diction
        "verity", "veracity", "candor", "candour", "forthright", "uprightness",
        "rectitude", "probity", "incorruptible", "unimpeachable",
        # Action-phrase patterns
        "speak plainly", "speak frankly", "put it plainly", "be frank",
        "must confess", "cannot in good conscience", "will not pretend",
        "refuse to deceive", "refuse to dissemble", "stand by the truth",
        "keep my word", "man of my word", "woman of my word",
        # Failure markers (signal P0/APY)
        "deceived", "lied", "misled", "concealed", "fabricated",
    ],
    "courage": [
        # Direct assertions
        "afraid", "scared", "brave", "bravery", "courage", "courageous",
        "risk", "risking", "hard to say", "nervous about", "terrified",
        "fear", "facing my fear",
        # Historical / formal diction
        "valiant", "valor", "valour", "fortitude", "dauntless", "intrepid",
        "undaunted", "bold", "boldness", "daring", "audacity", "audacious",
        "gallant", "gallantry", "resolute", "unflinching",
        # Action-phrase patterns
        "stood firm", "stood my ground", "held my ground", "did not flinch",
        "would not yield", "refused to flee", "pressed on", "pressed forward",
        "marched on", "in spite of danger", "at great personal risk",
        "knowing the cost", "despite the threat", "under fire",
        # Failure markers
        "cowardice", "fled", "retreated", "capitulated",
    ],
    "compassion": [
        # Direct assertions
        "care about", "worry about", "worrying about", "sad for", "feel for",
        "sorry for", "heart goes out", "feeling for", "concerned about them",
        # Historical / formal diction
        "mercy", "merciful", "tenderness", "charity", "charitable",
        "clemency", "benevolence", "benevolent", "forbearance", "pity",
        "empathy", "sympathize", "sympathy", "kindheartedness",
        "moved by", "touched by", "stricken by",
        # Action-phrase patterns
        "could not turn away", "could not ignore their suffering",
        "their pain is", "their suffering", "wept for", "wept with",
        "knelt beside", "sat with", "attended to",
        # Failure markers
        "indifferent", "unmoved", "turned away", "hardened my heart",
    ],
    "commitment": [
        # Direct assertions
        "promise", "promised", "commit", "committed", "dedicated", "dedication",
        "always will", "I will", "won't give up", "I won't stop", "I'll be there",
        # Historical / formal diction
        "vow", "vowed", "pledge", "pledged", "oath", "swore", "sworn",
        "bound by", "duty bound", "resolute", "steadfast", "unwavering",
        "unyielding", "persevere", "perseverance", "persist",
        # Action-phrase patterns
        "see it through", "to the end", "until it is done",
        "shall not rest", "will not rest", "cannot abandon",
        "will finish what", "gave my word", "gave my life to",
        "devoted myself", "devoted my life",
        # Failure markers
        "abandoned", "gave up", "broke my promise", "reneged",
    ],
    "patience": [
        # Direct assertions
        "patient", "patience", "waiting", "wait it out", "take time",
        "slow down", "in time", "eventually", "let it unfold",
        # Historical / formal diction
        "forbearance", "long-suffering", "endurance", "composure",
        "equanimity", "serenity", "temperate", "measured",
        # Action-phrase patterns
        "bide my time", "hold steady", "shall come", "will come in time",
        "not rushed", "without haste", "unhurried", "steady pace",
        "waited years", "waited decades", "endured the wait",
        # Failure markers
        "lost patience", "could wait no longer", "acted rashly",
    ],
    "responsibility": [
        # Direct assertions
        "my fault", "my responsibility", "responsible for", "should have",
        "accountable", "I owe", "I let", "I need to fix", "on me",
        # Historical / formal diction
        "duty", "obliged", "obligation", "culpable", "culpability",
        "answerable", "liable", "stewardship", "steward",
        # Action-phrase patterns
        "accept the blame", "accept responsibility", "bear the burden",
        "will answer for", "cannot shirk", "must answer", "cannot excuse",
        "falls to me", "it is my burden", "I bear responsibility",
        "failed in my duty", "neglected my duty",
        # Failure markers
        "shirked", "evaded", "deflected blame", "denied responsibility",
    ],
    "fairness": [
        # Direct assertions
        "fair", "fairness", "equal", "equality", "just", "justice",
        "deserves", "unfair", "unjust", "not right",
        # Historical / formal diction
        "equitable", "equity", "impartial", "impartiality", "unbiased",
        "even-handed", "proportionate", "righteous", "righteousness",
        "due process", "deserved", "merited", "what is owed",
        # Action-phrase patterns
        "treated equally", "judge by", "measure by", "regardless of",
        "no favoritism", "without distinction", "same standard",
        "denied justice", "robbed of justice",
        # Failure markers
        "prejudiced", "biased", "favored", "discriminated",
    ],
    "gratitude": [
        # Direct assertions
        "grateful", "gratitude", "thankful", "appreciate", "appreciation",
        "thank you", "lucky to have", "so glad", "means a lot",
        # Historical / formal diction
        "indebted", "beholden", "obliged", "grace", "blessing",
        "fortune", "fortunate", "count my blessings", "what I have been given",
        # Action-phrase patterns
        "owe a debt", "cannot repay", "will not forget",
        "will always remember", "have been given so much",
        "born into", "given the opportunity",
        # Failure markers
        "ungrateful", "took for granted", "never thanked",
    ],
    "curiosity": [
        # Direct assertions
        "wondering", "wonder", "curious", "curiosity", "want to know",
        "interested in", "explore", "fascinated", "trying to understand",
        # Historical / formal diction
        "inquire", "inquiry", "enquire", "inquisitive", "probe",
        "investigate", "investigation", "examine", "scrutinize",
        "study", "studied", "contemplated", "pondered",
        # Action-phrase patterns
        "must understand", "driven to know", "could not let it rest",
        "had to find out", "compelled to ask", "question everything",
        "ask why", "sought answers", "looked deeper",
        # Failure markers
        "incurious", "dismissed", "refused to question",
    ],
    "resilience": [
        # Direct assertions
        "keep going", "keep trying", "bounce back", "despite", "even though",
        "still going", "won't quit", "push through", "get through this",
        # Historical / formal diction
        "endure", "endured", "endurance", "persevere", "perseverance",
        "tenacity", "tenacious", "grit", "indomitable", "unyielding",
        "unconquered", "unbroken", "unbowed", "weathered",
        # Action-phrase patterns
        "rose again", "rose from", "rebuilt", "recovered from",
        "came back", "carried on", "pressed on despite",
        "refused to be broken", "would not break", "stood back up",
        "survived", "outlasted", "through hardship",
        # Failure markers
        "broke down", "gave in", "surrendered to despair",
    ],
    "love": [
        # Direct assertions
        "love", "I love", "care deeply", "cherish", "miss", "missing",
        "mean everything", "everything to me", "means the world",
        # Historical / formal diction
        "devotion", "devoted", "adore", "adoration", "affection",
        "tenderness", "ardor", "ardour", "reverence", "beloved",
        "dear to me", "dearest", "hold dear",
        # Action-phrase patterns
        "would give my life", "gave my life for", "sacrifice for",
        "cannot live without", "heart belongs to", "my heart",
        "bound to", "united with", "would not leave",
        # Failure markers
        "abandoned", "forsook", "betrayed those who loved",
    ],
    "growth": [
        # Direct assertions
        "better at", "getting better", "improve", "improving", "learning",
        "growing", "want to be", "trying to become", "working on myself",
        # Historical / formal diction
        "self-improvement", "cultivation", "develop", "development",
        "educate myself", "enlighten", "progress", "advancement",
        "matured", "refined", "discipline", "disciplined",
        # Action-phrase patterns
        "became a better", "learned from", "that experience taught me",
        "changed my thinking", "altered my view", "reconsidered",
        "came to understand", "deepened my understanding",
        "no longer believe", "had to unlearn",
        # Failure markers
        "stagnated", "refused to change", "too proud to learn",
    ],
    "independence": [
        # Direct assertions
        "on my own", "by myself", "my choice", "my decision", "self-reliant",
        "don't need anyone", "figure it out myself",
        # Historical / formal diction
        "autonomy", "autonomous", "sovereignty", "sovereign",
        "self-determination", "self-sufficient", "free to choose",
        "independent", "independence", "liberty", "liberated",
        "not beholden", "free from", "owe nothing to",
        # Action-phrase patterns
        "will not be told", "answer to no one", "chose for myself",
        "made my own way", "carved my own path", "refused to submit",
        "would not be controlled", "on my own terms",
        # Failure markers
        "subjugated", "surrendered my autonomy", "dependent on",
    ],
    "loyalty": [
        # Direct assertions
        "always there", "stand by", "standing by", "loyal", "loyalty",
        "won't leave", "I'll be there", "through thick and thin",
        # Historical / formal diction
        "fidelity", "faithful", "steadfast", "allegiance", "fealty",
        "devoted", "devotion", "true to", "honor bound", "solidarity",
        # Action-phrase patterns
        "stayed when others left", "would not abandon",
        "stood with", "remain at", "never wavered in my support",
        "did not desert", "held the line", "kept faith with",
        "did not betray",
        # Failure markers
        "betrayed", "deserted", "defected", "turned on",
    ],
    "humility": [
        # Direct assertions
        "I was wrong", "my mistake", "I made a mistake", "I learned",
        "I missed", "I didn't know", "I was mistaken", "I need to admit",
        # Historical / formal diction
        "erred", "error of judgment", "concede", "acknowledge",
        "must own", "own this failure", "modest", "modesty",
        "not above", "no better than", "no greater than",
        # Action-phrase patterns
        "I have no right to", "who am I to", "I cannot claim",
        "credit belongs to", "not my doing", "I was given",
        "I owe that to", "others did more", "stood on the shoulders",
        "I cannot pretend", "I must confess my error",
        # Failure markers
        "arrogance", "arrogant", "refused to admit", "could not concede",
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
