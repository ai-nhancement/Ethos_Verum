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

import logging
import re
import time
from core.apy_patterns import APY_PRESSURE_RE as _APY_PRESSURE_RE_INGEST
from typing import Dict, List

_log = logging.getLogger(__name__)

from core.config import get_config
from core.document_store import get_document_store
from core.polarity_layer import detect_polarity
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
# Negation detection — suppress signals where a positive keyword is negated
# ---------------------------------------------------------------------------

# Negation words that appear in a tight pre-keyword window (~30 chars, ~4-5 words).
# If found immediately before the keyword, the signal is suppressed.
# Example: "I was not brave" → "not" before "brave" → courage signal suppressed.
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no(?:\s+longer)?|without|lacks?|lacked|lacking|"
    r"fails?\s+to|failed\s+to|unable\s+to|"
    r"can'?t|won'?t|didn'?t|doesn'?t|wasn'?t|"
    r"weren'?t|isn'?t|aren'?t|couldn'?t|wouldn'?t|"
    r"hardly|barely|scarcely|seldom|rarely)\b",
    re.IGNORECASE,
)

# Negating prefixes that attach directly to keyword characters (e.g. "dis" in "dishonest").
_NEGATION_PREFIX_RE = re.compile(r"(?:dis|un|in|ir|im|non)-?$", re.IGNORECASE)

# Keywords that are failure/violation markers — their presence in a passage
# signals P0, not P1. The negation check is intentionally SKIPPED for these:
# when a failure keyword is preceded by negation ("never fled", "did not betray"),
# it becomes positive evidence — the value held.
_FAILURE_MARKER_KWS: frozenset = frozenset({
    "deceived", "lied", "misled", "concealed", "fabricated",
    "cowardice", "fled", "retreated", "capitulated",
    "indifferent", "unmoved", "turned away", "hardened my heart",
    "abandoned", "gave up", "broke my promise", "reneged",
    "lost patience", "could wait no longer", "acted rashly",
    "shirked", "evaded", "deflected blame", "denied responsibility",
    "prejudiced", "biased", "favored", "discriminated",
    "ungrateful", "took for granted", "never thanked",
    "incurious", "dismissed", "refused to question",
    "broke down", "gave in", "surrendered to despair",
    "betrayed", "deserted", "defected", "turned on",
    "arrogance", "arrogant", "refused to admit", "could not concede",
    "stagnated", "refused to change", "too proud to learn",
    "subjugated", "surrendered my autonomy",
})


# ---------------------------------------------------------------------------
# Semantic layer helpers
# ---------------------------------------------------------------------------

def _semantic_signals(
    text: str,
    significance: float,
    cfg,
) -> List[Dict]:
    """
    Run semantic prototype matching against the passage (direction-aware).

    Queries both the hold prototype collection (value demonstrated) and the
    failure prototype collection (value violated). A signal is only emitted
    when the hold score exceeds the failure score for that value — preventing
    violation passages from contributing positive evidence.

    When failure prototypes are not yet built, falls back to hold-only behavior.

    Confidence = hold_score - (failure_score * 0.5), scaled by semantic_weight.
    Returns [] if semantic store is unavailable or disabled.
    """
    if not cfg.semantic_enabled:
        return []
    try:
        from core.embedder import encode, is_available as emb_ok
        from core.semantic_store import get_semantic_store
        if not emb_ok():
            return []
        # Archaic preprocessing before embedding (Phase 3)
        try:
            from core.temporal_layer import preprocess_archaic
            embed_text = preprocess_archaic(text)
        except Exception:
            embed_text = text
        vec = encode(embed_text)
        if vec is None:
            return []
        store = get_semantic_store()

        # Hold prototypes: value being demonstrated
        hold_hits = store.query_passage(
            vec,
            top_k=cfg.semantic_top_k,
            score_threshold=cfg.semantic_threshold,
        )

        # Failure prototypes: value being violated.
        # Use same threshold so only meaningful failure matches are considered.
        failure_scores: Dict[str, float] = {}
        if store.failure_prototypes_ready():
            for vname, fscore in store.query_failure_passage(
                vec,
                top_k=cfg.semantic_top_k,
                score_threshold=cfg.semantic_threshold,
            ):
                failure_scores[vname] = fscore

        results = []
        excerpt = text[:150] if len(text) <= 150 else text[:147] + "..."
        for value_name, hold_score in hold_hits:
            fail_score = failure_scores.get(value_name, 0.0)
            # Suppress: passage resembles a violation more than a demonstration
            if fail_score >= hold_score:
                continue
            # Net confidence: hold lead minus failure discount
            net_score = hold_score - (fail_score * 0.5)
            results.append({
                "value_name": value_name,
                "text_excerpt": excerpt,
                "significance": significance,
                "disambiguation_confidence": round(float(net_score) * cfg.semantic_weight, 4),
                "source": "semantic",
            })
        return results
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Layer 1b helper — lexicon
# ---------------------------------------------------------------------------

def _lexicon_signals(
    text: str,
    significance: float,
    doc_type: str,
    cfg,
) -> List[Dict]:
    """
    Run MFD2.0 + MoralStrength lexicon matching on a passage.
    Returns all signals (virtue and vice) — caller filters by polarity.
    """
    if not cfg.lexicon_enabled:
        return []
    try:
        from core.lexicon_layer import lexicon_signals, is_available
        if not is_available():
            return []
        return lexicon_signals(text, significance, doc_type)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Phrase layer helper — compositional verb + vice-word detection
# ---------------------------------------------------------------------------

def _phrase_signals(
    text: str,
    significance: float,
    doc_type: str,
    cfg,
) -> tuple:
    """
    Run phrase composition detection on a passage.

    Returns (signals, consumed_tokens) where:
      signals          — phrase-level signal dicts with 'polarity_hint' key
      consumed_tokens  — lowercase token strings consumed by phrase matches
    """
    if not cfg.lexicon_enabled:
        return [], set()
    try:
        from core.phrase_layer import phrase_signals
        return phrase_signals(text, significance, doc_type)
    except Exception:
        return [], set()


# ---------------------------------------------------------------------------
# Layer 3 helper
# ---------------------------------------------------------------------------

def _layer3_signals(
    text: str,
    significance: float,
    doc_type: str,
    already_detected: List[Dict],
    cfg,
) -> tuple:
    """
    Run Layer 3 (structural + zero-shot) on a passage.

    Returns (structural_score: float, new_signals: List[Dict], zs_agreement: Dict[str, float]).
      structural_score — value-agnostic adversity score; caller boosts all signals.
      new_signals      — zero-shot standalone detections (not found by L1/L2).
      zs_agreement     — {value_name: zs_confidence} for values in already_detected
                         that zero-shot also confirmed at >= zeroshot_threshold.
                         Caller applies agreement boost to those signals.
    """
    try:
        from core.structural_layer import layer3_signals
        candidate_values = [s["value_name"] for s in already_detected]
        return layer3_signals(
            text=text,
            significance=significance,
            doc_type=doc_type,
            candidate_values=candidate_values,
            zeroshot_threshold=cfg.zeroshot_threshold,
            zeroshot_standalone_threshold=cfg.zeroshot_standalone_threshold,
            zeroshot_enabled=cfg.zeroshot_enabled,
        )
    except Exception:
        return 0.0, [], {}


# ---------------------------------------------------------------------------
# Layer 3c helper — MFT classifier
# ---------------------------------------------------------------------------

def _mft_signals(text: str, cfg) -> dict:
    """
    Run MFT classifier on the passage.
    Returns the mft_classifier.mft_signals() dict (boosted_values, vice_flags).
    Fail-open: returns empty dict on any error or when disabled.
    """
    try:
        from core.mft_classifier import mft_signals
        return mft_signals(
            text,
            min_virtue_score=cfg.mft_min_virtue_score,
            min_vice_score=cfg.mft_min_vice_score,
        )
    except Exception:
        return {"boosted_values": [], "vice_flags": []}


# ---------------------------------------------------------------------------
# Disambiguation — §7.7
# ---------------------------------------------------------------------------

# Compiled regex for first-person pronouns.
_FIRST_PERSON_RE = re.compile(r"\b(I|me|my|myself|we|our|us)\b", re.IGNORECASE)

# Values where a first-person pronoun must appear within ±80 chars of the
# keyword. Values NOT in this set (compassion, fairness, gratitude, love,
# loyalty) can legitimately refer to a third party in biographical text.
_REQUIRES_FIRST_PERSON: set = {
    "courage",
    "commitment",
    "patience",
    "responsibility",
    "curiosity",
    "resilience",
    "growth",
    "independence",
    "humility",
}

# Per-value context-window disqualifiers.  If the pattern matches the 160-char
# window around the keyword, the signal is dropped regardless of first-person.
_DISQUALIFIERS: Dict[str, re.Pattern] = {
    # "my patient recovered" / "the patient was" -- medical noun, not the virtue
    "patience": re.compile(
        r"\b(?:my|the|his|her|their|a|our)\s+patients?\s+\b",
        re.IGNORECASE,
    ),
    # "fair weather", "fair hair", "fair trade" -- adjective, not the value
    "fairness": re.compile(
        r"\bfair\s+(?:weather|wind|sky|skies|hair|skin|complexion|use|trade|market)\b",
        re.IGNORECASE,
    ),
    # "to be honest though / with you" -- filler phrase, not an integrity claim
    "integrity": re.compile(
        r"\bto\s+be\s+honest\s*(?:,|\bthough\b|\bwith\s+you\b|\bI\s+don['\u2019]t\b|\babout\s+my\s+(?:schedule|day|week|plans?)\b)",
        re.IGNORECASE,
    ),
    # "devoted time/energy/attention" -- effort allocation, not loyalty
    "loyalty": re.compile(
        r"\bdevoted?\s+(?:\w+\s+){0,2}(?:time|hours|energy|attention|resources|effort|efforts)\b",
        re.IGNORECASE,
    ),
    # "love pizza/coffee/sports/movies/music" -- preference, not bond
    "love": re.compile(
        r"\b(?:I\s+)?loves?\s+(?:pizza|coffee|tea|beer|wine|food|sports?|movies?|music|games?|this|that|it|them)\b",
        re.IGNORECASE,
    ),
    # "survived/recovered from surgery/cancer/treatment" -- medical, not moral resilience
    "resilience": re.compile(
        r"\b(?:survived?|recovered?)\s+(?:from\s+)?(?:surgery|cancer|the\s+operation|chemotherapy|"
        r"the\s+procedure|the\s+hospital|the\s+illness|the\s+disease|the\s+infection|the\s+treatment)\b",
        re.IGNORECASE,
    ),
    # "I will call/text/email/meet you" / "committed to targets" / "dedicated to gym" -- not moral commitment
    "commitment": re.compile(
        r"\bI\s+will\s+(?:call|text|email|send|meet|see\s+you|be\s+there\s+at|arrive|"
        r"attend\s+the|join\s+the|check|look\s+into)\b"
        r"|\bcommitted\s+to\s+(?:the\s+)?(?:project|plan|goal|target|objective|"
        r"budget|timeline|schedule|roadmap|strategy|kpi|okr)\b"
        r"|\bcommitted\s+to\s+(?:achieving|reaching|hitting|meeting|delivering|"
        r"executing|fulfilling|implementing|completing)\b"
        r"|\bdedicated\s+to\s+(?:the\s+)?(?:gym|fitness|working\s+out|diet|exercise|routine)\b",
        re.IGNORECASE,
    ),
    # "responsible for the project/meeting/report" -- task assignment, not moral accountability
    "responsibility": re.compile(
        r"\bresponsible\s+for\s+(?:the\s+)?(?:project|meeting|report|presentation|"
        r"event|campaign|website|design|scheduling|organizing|managing|coordinating)\b",
        re.IGNORECASE,
    ),
    # "thanks for joining/attending" -- courtesy, not deep gratitude
    "gratitude": re.compile(
        r"\bthanks?\s+(?:you\s+)?for\s+(?:joining|attending|coming|being\s+here|"
        r"your\s+time|your\s+participation|your\s+presence|tuning\s+in)\b",
        re.IGNORECASE,
    ),
    # "interested in the position/role/job" -- professional interest, not intellectual curiosity
    "curiosity": re.compile(
        r"\binterested\s+in\s+(?:the\s+)?(?:position|role|opportunity|job|opening|vacancy)\b",
        re.IGNORECASE,
    ),
    # "not above the law/average/sea level" -- common idiom, not moral humility
    "humility": re.compile(
        r"\bnot\s+above\s+(?:the\s+)?(?:law|average|minimum|maximum|sea\s+level|ground|the\s+rules)\b",
        re.IGNORECASE,
    ),
    # "put on a brave face" -- mask/performance, not genuine courage
    "courage": re.compile(
        r"\bbrave\s+face\b",
        re.IGNORECASE,
    ),
    # "moved by the performance/film/music" -- aesthetic emotion, not compassion for suffering
    "compassion": re.compile(
        r"\bmoved\s+by\s+(?:the\s+)?(?:performance|film|movie|music|song|score|concert|play|book|story)\b",
        re.IGNORECASE,
    ),
    # "getting better from the flu" / "improving my score" -- medical/metric, not character growth
    "growth": re.compile(
        r"\bgetting\s+better\s+(?:from|after|following)\b"
        r"|\b(?:improved|improving)\s+(?:my\s+)?(?:cholesterol|blood\s+pressure|"
        r"symptoms|condition|score|grade|ranking|rating|numbers?)\b",
        re.IGNORECASE,
    ),

    # "on my own to the store" / "cooking by myself" -- trivial self-sufficiency, not the value
    "independence": re.compile(
        r"\b(?:on\s+my\s+own|by\s+myself)\s+"
        r"(?:to\s+(?:the|a)\s+\w+|at\s+(?:the|a)\s+\w+|"
        r"shopping|cooking|driving|walking|cleaning|playing|running|working\s+out)\b"
        r"|\b(?:shopping|cooking|driving|walking|cleaning|playing|running|working\s+out)"
        r"\s+(?:on\s+my\s+own|by\s+myself)\b",
        re.IGNORECASE,
    ),
}



def _check_signal(
    text_lower: str,
    value_name: str,
    kw: str,
    match_idx: int,
    doc_type: str = "unknown",
):
    """
    Returns (is_valid: bool, confidence: float) for a keyword hit.

    Check 1 - per-value disqualifier on 160-char context window (drops outright).
    Check 2 - first-person proximity for values in _REQUIRES_FIRST_PERSON;
              bypassed when doc_type == 'action' (biographical text, third-person valid).

    Confidence:
      1.0  first-person pronoun found in context window (strong self-attribution)
      0.7  action doc, first-person not required (biographical third-person)
      0.6  non-required value, no first-person found (weak but accepted)

    Never raises; fail-open returns (True, 1.0).
    """
    try:
        ctx_start = max(0, match_idx - 80)
        ctx_end   = min(len(text_lower), match_idx + len(kw) + 80)
        ctx       = text_lower[ctx_start:ctx_end]

        # Check 0: negation — suppress when a positive keyword is immediately
        # preceded by a negation word or negating prefix.
        # Skipped for failure-marker keywords: negated failures ("never fled",
        # "did not betray") are positive evidence and should not be suppressed.
        kw_lower = kw.lower()
        if kw_lower not in _FAILURE_MARKER_KWS:
            # (a) Negating prefix directly before keyword characters — e.g. "dishonest"
            if match_idx > 0:
                prefix_window = text_lower[max(0, match_idx - 4):match_idx]
                if _NEGATION_PREFIX_RE.search(prefix_window):
                    return False, 0.0

            # (b) Negation word in tight 30-char pre-keyword window.
            #     Skipped when the keyword itself contains structural negation
            #     ("won't give up", "did not yield") — those are positive signals.
            if not any(s in kw_lower for s in ("not", "n't", "never", "without", "cannot")):
                pre_window = text_lower[max(0, match_idx - 30):match_idx]
                if _NEGATION_RE.search(pre_window):
                    return False, 0.0

        # Check 1: disqualifier -- overlap-based: only block if the disqualifier
        # match overlaps with the actual keyword position (not just nearby text).
        disq = _DISQUALIFIERS.get(value_name)
        if disq:
            kw_end = match_idx + len(kw)
            for m in disq.finditer(text_lower):
                if m.start() < kw_end and m.end() > match_idx:
                    return False, 0.0

        # Check 2: first-person proximity
        is_action = (doc_type or "").lower().strip() == "action"
        has_fp    = bool(_FIRST_PERSON_RE.search(ctx))

        if value_name in _REQUIRES_FIRST_PERSON:
            if not has_fp and not is_action:
                return False, 0.0
            conf = 1.0 if has_fp else 0.7   # action doc bypass: 0.7
        else:
            conf = 1.0 if has_fp else 0.6   # third-person accepted, lower confidence

        return True, conf

    except Exception:
        return True, 1.0  # fail-open



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
        _log.exception("process_figure failed for %s", session_id)
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
    _passage_idx = 0
    _apy_window_n = cfg.apy_context_window_n

    for row in passages[:_MAX_PASSAGES_PER_RUN]:
        text = str(row["text"])
        significance = float(row["significance"])
        doc_type = str(row["doc_type"])
        ts = float(row["ts"])
        record_id = str(row["id"])
        # Phase 3 — temporal/translation calibration
        source_auth  = float(row["source_authenticity"]) if row["source_authenticity"] is not None else 1.0
        pub_year_val = row["pub_year"]  # may be None

        if significance < cfg.min_significance_threshold or not text:
            latest_ts = max(latest_ts, ts)
            continue

        # Keyword signals (Layer 1)
        kw_signals = extract_value_signals(text, record_id, significance, doc_type)
        for s in kw_signals:
            s["source"] = "keyword"

        # Lexicon signals (Layer 1b) — MFD2.0 + MoralStrength virtue matches
        lex_signals: List[Dict] = []
        if cfg.lexicon_enabled:
            lex_signals = _lexicon_signals(text, significance, doc_type, cfg)

        # Phrase layer (Layer 1c) — compositional verb + vice-word detection
        # Emits signals with pre-determined polarity_hint; bypasses Tier 1/2 detection.
        _p_sigs, _consumed_vice = _phrase_signals(text, significance, doc_type, cfg)

        # Semantic signals (Layer 2) — only if semantic layer available
        sem_signals = _semantic_signals(text, significance, cfg)

        # Merge L1 + L1b (lexicon): boost keyword signal when lexicon agrees;
        # add lexicon-only virtue signals if above standalone threshold.
        kw_values = {s["value_name"] for s in kw_signals}
        merged: List[Dict] = list(kw_signals)
        for lex in lex_signals:
            if lex["lexicon_polarity"] != "virtue":
                continue  # vice signals not merged into main signal list
            if lex["value_name"] in kw_values:
                for kw in merged:
                    if kw["value_name"] == lex["value_name"]:
                        boosted = min(1.0, kw["disambiguation_confidence"]
                                      + lex["disambiguation_confidence"] * 0.15)
                        kw["disambiguation_confidence"] = round(boosted, 4)
                        if "lexicon" not in kw["source"]:
                            kw["source"] = kw["source"] + "+lexicon"
            elif lex["disambiguation_confidence"] >= cfg.lexicon_standalone_min_conf:
                sig = {k: v for k, v in lex.items() if k != "lexicon_polarity"}
                merged.append(sig)

        # Phrase signals (Layer 1c): add standalone phrase-level detections.
        # Each has a pre-determined polarity_hint; polarity detection is skipped
        # at recording time for these signals.
        for ps in _p_sigs:
            merged.append({k: v for k, v in ps.items()})

        # Merge L1+L1b + L2 (semantic): boost when both agree.
        merged_values = {s["value_name"] for s in merged}
        for sem in sem_signals:
            if sem["value_name"] in merged_values:
                for s in merged:
                    if s["value_name"] == sem["value_name"]:
                        boosted = min(1.0, s["disambiguation_confidence"]
                                      + sem["disambiguation_confidence"] * 0.20)
                        s["disambiguation_confidence"] = round(boosted, 4)
                        if "semantic" not in s["source"]:
                            s["source"] = s["source"] + "+semantic"
            else:
                merged.append(sem)

        # Layer 3 — structural patterns + zero-shot entailment
        struct_score = 0.0
        if cfg.layer3_enabled:
            struct_score, l3_new, zs_agreement = _layer3_signals(text, significance,
                                                                  doc_type, merged, cfg)
            # Boost confidence of existing signals with structural evidence
            if struct_score > 0.0:
                boost = round(struct_score * cfg.structural_resistance_boost, 4)
                for s in merged:
                    s["disambiguation_confidence"] = round(
                        min(1.0, s["disambiguation_confidence"] + boost), 4
                    )
                    if "structural" not in s["source"]:
                        s["source"] = s["source"] + "+structural"
            # Zero-shot agreement boost for signals already detected by L1/L2
            for s in merged:
                if s["value_name"] in zs_agreement:
                    zs_conf = zs_agreement[s["value_name"]]
                    extra = cfg.zeroshot_agreement_boost * zs_conf
                    s["disambiguation_confidence"] = round(
                        min(1.0, s["disambiguation_confidence"] + extra), 4
                    )
                    if "zeroshot" not in s["source"]:
                        s["source"] = s["source"] + "+zeroshot"
            # New zero-shot detections not found by L1/L2
            merged.extend(l3_new)

        # Layer 3c — MFT classifier
        if cfg.mft_enabled:
            mft = _mft_signals(text, cfg)
            merged_values = {s["value_name"] for s in merged}
            for hint in mft["boosted_values"]:
                vname = hint["value_name"]
                score = hint["score"]
                if vname in merged_values:
                    boost = round(score * cfg.mft_agreement_boost, 4)
                    for s in merged:
                        if s["value_name"] == vname:
                            s["disambiguation_confidence"] = round(
                                min(1.0, s["disambiguation_confidence"] + boost), 4
                            )
                            if "mft" not in s["source"]:
                                s["source"] = s["source"] + "+mft"
                elif score >= cfg.mft_standalone_threshold:
                    merged.append({
                        "value_name":               vname,
                        "text_excerpt":             text[:200],
                        "significance":             significance,
                        "disambiguation_confidence": round(score * cfg.mft_standalone_weight, 4),
                        "source":                   "mft",
                    })
                    merged_values.add(vname)

        # Phase 3 — apply temporal/translation calibration to all signal confidences
        if source_auth < 1.0 or pub_year_val is not None:
            from core.temporal_layer import calibrate_confidence, pub_year_discount
            yr_discount = pub_year_discount(pub_year_val)
            for s in merged:
                s["disambiguation_confidence"] = calibrate_confidence(
                    s["disambiguation_confidence"], source_auth, yr_discount
                )

        # Resistance is passage-scoped (depends only on text/significance/doc_type),
        # so compute once and skip the entire passage if below threshold.
        resistance = compute_resistance(text, significance, doc_type)

        # Per-signal resistance modulation: when no structural adversity is detected
        # (struct_score == 0.0), apply a 15% discount to all signals in this passage.
        # Values mentioned in low-adversity contexts receive proportionally lower
        # resistance credit than values mentioned in contexts with structural adversity.
        # Scale: resistance × (0.85 + 0.15 × struct_score) → [0.85r, 1.00r]
        struct_resistance_factor = round(0.85 + 0.15 * struct_score, 4)

        # Build per-value lexicon vice/virtue score maps for polarity detection.
        # These are used by Tier 2 of detect_polarity() to confirm or override
        # the target-word proximity result.
        _lex_vice_scores:    Dict[str, float] = {}
        _lex_virtue_scores:  Dict[str, float] = {}
        for _ls in lex_signals:
            _vn   = _ls.get("value_name", "")
            _conf = float(_ls.get("disambiguation_confidence", 0.0))
            if _ls.get("lexicon_polarity") == "vice":
                _lex_vice_scores[_vn]   = max(_lex_vice_scores.get(_vn, 0.0), _conf)
            elif _ls.get("lexicon_polarity") == "virtue":
                _lex_virtue_scores[_vn] = max(_lex_virtue_scores.get(_vn, 0.0), _conf)

        if resistance >= cfg.min_resistance_threshold:
            for sig in merged:
                sig_resistance = round(resistance * struct_resistance_factor, 4)
                vname = sig["value_name"]
                # Per-signal polarity: phrase-layer signals carry a pre-determined
                # polarity_hint that bypasses Tier 1/2 detection.  All other signals
                # run the full detect_polarity() path.
                if "polarity_hint" in sig:
                    polarity  = sig["polarity_hint"]
                    pol_conf  = sig["disambiguation_confidence"]
                else:
                    # match_idx=None for non-keyword signals (semantic, zero-shot, MFT)
                    # causes polarity Tier 1 to search the full passage rather than
                    # a window anchored at position 0, which would be meaningless.
                    polarity, pol_conf = detect_polarity(
                        text=text,
                        match_idx=sig.get("match_idx"),  # None when key absent
                        value_name=vname,
                        lexicon_vice_score=_lex_vice_scores.get(vname, 0.0),
                        lexicon_virtue_score=_lex_virtue_scores.get(vname, 0.0),
                        cfg=cfg,
                    )
                val_store.record_observation(
                    session_id=session_id,
                    turn_id=record_id,
                    record_id=record_id,
                    ts=ts,
                    value_name=vname,
                    text_excerpt=sig["text_excerpt"],
                    significance=significance,
                    resistance=sig_resistance,
                    disambiguation_confidence=sig.get("disambiguation_confidence", 1.0),
                    doc_type=doc_type,
                    value_polarity=polarity,
                    polarity_confidence=pol_conf,
                )
                val_store.upsert_registry(
                    session_id=session_id,
                    value_name=sig["value_name"],
                    significance=significance,
                    resistance=sig_resistance,
                    ts=ts,
                    doc_type=doc_type,
                )
                # Cross-figure aggregate (session_id='')
                val_store.upsert_registry(
                    session_id="",
                    value_name=sig["value_name"],
                    significance=significance,
                    resistance=sig_resistance,
                    ts=ts,
                    doc_type=doc_type,
                )
                recorded += 1

        # Write APY pressure context if this passage has pressure markers
        # (independent of resistance — APY context is useful even for weak passages)
        apy_markers = [m.group(0).lower() for m in _APY_PRESSURE_RE_INGEST.finditer(text)]
        if apy_markers:
            val_store.write_apy_context(
                session_id=session_id,
                record_id=record_id,
                ts=ts,
                passage_idx=_passage_idx,
                markers=', '.join(apy_markers),
                window_n=_apy_window_n,
            )
        _passage_idx += 1
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
    doc_type: str = "unknown",
) -> List[Dict]:
    """
    Keyword-scan text against VALUE_VOCAB with disambiguation filter.
    Returns a list of {value_name, text_excerpt, significance, disambiguation_confidence} dicts.
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
            idx = text_lower.find(kw.lower())
            if idx < 0:
                continue
            valid, conf = _check_signal(text_lower, value_name, kw, idx, doc_type)
            if not valid:
                continue  # try next keyword for same value before giving up
            excerpt = _extract_excerpt(text, kw, max_len=150)
            results.append({
                "value_name": value_name,
                "text_excerpt": excerpt,
                "significance": significance,
                "disambiguation_confidence": conf,
                "match_idx": idx,          # character position for polarity window
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
