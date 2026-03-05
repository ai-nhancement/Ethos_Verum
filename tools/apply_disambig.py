"""Apply all disambiguation improvements to value_extractor.py."""
import sys
sys.path.insert(0, 'C:/Ethos')

with open('C:/Ethos/core/value_extractor.py', encoding='utf-8') as f:
    src = f.read()

# ── CHANGE 1: Replace _DISQUALIFIERS dict with expanded version ──────────────
old_disq_start = src.index('# Per-value context-window disqualifiers.')
old_disq_end   = src.index('\ndef _is_valid_signal(')

new_disq = r"""# Per-value context-window disqualifiers.  If the pattern matches the 160-char
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
        r"\b(?:I\s+)?love\s+(?:pizza|coffee|tea|beer|wine|food|sports?|movies?|music|games?|this|that|it|them)\b",
        re.IGNORECASE,
    ),
    # "survived/recovered from surgery/cancer/treatment" -- medical, not moral resilience
    "resilience": re.compile(
        r"\b(?:survived?|recovered?)\s+(?:from\s+)?(?:surgery|cancer|the\s+operation|chemotherapy|"
        r"the\s+procedure|the\s+hospital|the\s+illness|the\s+disease|the\s+infection|the\s+treatment)\b",
        re.IGNORECASE,
    ),
    # "I will call/text/email/meet you" -- scheduling, not moral commitment
    "commitment": re.compile(
        r"\bI\s+will\s+(?:call|text|email|send|meet|see\s+you|be\s+there\s+at|arrive|"
        r"attend\s+the|join\s+the|check|look\s+into)\b",
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
}

"""

src = src[:old_disq_start] + new_disq + src[old_disq_end:]

# ── CHANGE 2: Replace _is_valid_signal() with _check_signal() ────────────────
old_fn_start = src.index('\ndef _is_valid_signal(')
old_fn_end   = src.index('\n\n# ---------------------------------------------------------------------------\n# Public entry point')

new_fn = """
def _check_signal(
    text_lower: str,
    value_name: str,
    kw: str,
    match_idx: int,
    doc_type: str = "unknown",
):
    \"\"\"
    Returns (is_valid: bool, confidence: float) for a keyword hit.

    Check 1 - per-value disqualifier on 160-char context window (drops outright).
    Check 2 - first-person proximity for values in _REQUIRES_FIRST_PERSON;
              bypassed when doc_type == 'action' (biographical text, third-person valid).

    Confidence:
      1.0  first-person pronoun found in context window (strong self-attribution)
      0.7  action doc, first-person not required (biographical third-person)
      0.6  non-required value, no first-person found (weak but accepted)

    Never raises; fail-open returns (True, 1.0).
    \"\"\"
    try:
        ctx_start = max(0, match_idx - 80)
        ctx_end   = min(len(text_lower), match_idx + len(kw) + 80)
        ctx       = text_lower[ctx_start:ctx_end]

        # Check 1: disqualifier
        disq = _DISQUALIFIERS.get(value_name)
        if disq and disq.search(ctx):
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

"""
src = src[:old_fn_start] + new_fn + src[old_fn_end:]

# ── CHANGE 3: Update extract_value_signals() signature ───────────────────────
src = src.replace(
    'def extract_value_signals(\n    text: str,\n    record_id: str,\n    significance: float,\n) -> List[Dict]:',
    'def extract_value_signals(\n    text: str,\n    record_id: str,\n    significance: float,\n    doc_type: str = "unknown",\n) -> List[Dict]:'
)
src = src.replace(
    '    Keyword-scan text against VALUE_VOCAB (case-insensitive substring).\n    Returns a list of {value_name, text_excerpt, significance} dicts.',
    '    Keyword-scan text against VALUE_VOCAB with disambiguation filter.\n    Returns a list of {value_name, text_excerpt, significance, disambiguation_confidence} dicts.'
)

# ── CHANGE 3b: Update the loop body ──────────────────────────────────────────
src = src.replace(
    '            if not _is_valid_signal(text_lower, value_name, kw, idx):\n'
    '                continue  # try next keyword for same value before giving up\n'
    '            excerpt = _extract_excerpt(text, kw, max_len=150)\n'
    '            results.append({\n'
    '                "value_name": value_name,\n'
    '                "text_excerpt": excerpt,\n'
    '                "significance": significance,\n'
    '                "disambiguation_confidence": 1.0,\n'
    '            })',
    '            valid, conf = _check_signal(text_lower, value_name, kw, idx, doc_type)\n'
    '            if not valid:\n'
    '                continue  # try next keyword for same value before giving up\n'
    '            excerpt = _extract_excerpt(text, kw, max_len=150)\n'
    '            results.append({\n'
    '                "value_name": value_name,\n'
    '                "text_excerpt": excerpt,\n'
    '                "significance": significance,\n'
    '                "disambiguation_confidence": conf,\n'
    '            })'
)

# ── CHANGE 4: Pass doc_type into extract_value_signals ───────────────────────
src = src.replace(
    '        signals = extract_value_signals(text, record_id, significance)',
    '        signals = extract_value_signals(text, record_id, significance, doc_type)'
)

with open('C:/Ethos/core/value_extractor.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("All 4 changes applied.")
