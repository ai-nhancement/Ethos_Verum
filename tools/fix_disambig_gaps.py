"""
Close remaining disambiguation gaps:
  1. Add growth disqualifier (medical recovery, skill-score improvement)
  2. Add independence disqualifier (trivial self-sufficiency)
  3. Expand commitment disqualifier (corporate goals, gym dedication)
"""

with open('C:/Ethos/core/value_extractor.py', encoding='utf-8') as f:
    src = f.read()

# ── GAP 1: growth disqualifier ────────────────────────────────────────────────
# False positives:
#   "getting better from the flu"       (medical recovery, not values-level growth)
#   "improving my cholesterol/score"    (metric improvement, not character)
# Insert before the closing } of _DISQUALIFIERS (before the lone } + blank line + def _check_signal)
old_closing = '}\n\n\ndef _check_signal('
new_entries = ''',
    # "getting better from the flu" / "improving my score" -- medical/metric, not character growth
    "growth": re.compile(
        r"\\bgetting\\s+better\\s+(?:from|after|following)\\b"
        r"|\\b(?:improved|improving)\\s+(?:my\\s+)?(?:cholesterol|blood\\s+pressure|"
        r"symptoms|condition|score|grade|ranking|rating|numbers?)\\b",
        re.IGNORECASE,
    ),
    # "on my own to the store" / "by myself shopping" -- trivial, not independence-as-value
    "independence": re.compile(
        r"\\b(?:on\\s+my\\s+own|by\\s+myself)\\s+"
        r"(?:to\\s+(?:the|a)\\s+\\w+|at\\s+(?:the|a)\\s+\\w+|"
        r"shopping|cooking|driving|walking|cleaning|playing|running|working\\s+out)\\b",
        re.IGNORECASE,
    ),
}

'''

# Replace the lone closing } before _check_signal
assert src.count(old_closing) == 1, f"Expected 1 match, got {src.count(old_closing)}"
src = src.replace(old_closing, new_entries + '\n\ndef _check_signal(')

# ── GAP 3: Expand commitment disqualifier ─────────────────────────────────────
# Current pattern only catches scheduling ("I will call/text/email...").
# Additional false positives:
#   "committed to achieving our Q4 targets"   (corporate/OKR language)
#   "dedicated to the gym / fitness"          (routine habit, not moral pledge)
old_commitment = (
    '    # "I will call/text/email/meet you" -- scheduling, not moral commitment\n'
    '    "commitment": re.compile(\n'
    '        r"\\bI\\s+will\\s+(?:call|text|email|send|meet|see\\s+you|be\\s+there\\s+at|arrive|"\n'
    '        r"attend\\s+the|join\\s+the|check|look\\s+into)\\b",\n'
    '        re.IGNORECASE,\n'
    '    ),'
)
new_commitment = (
    '    # "I will call/text/email/meet you" / "committed to targets" / "dedicated to gym" -- not moral commitment\n'
    '    "commitment": re.compile(\n'
    '        r"\\bI\\s+will\\s+(?:call|text|email|send|meet|see\\s+you|be\\s+there\\s+at|arrive|"\n'
    '        r"attend\\s+the|join\\s+the|check|look\\s+into)\\b"\n'
    '        r"|\\bcommitted\\s+to\\s+(?:the\\s+)?(?:project|plan|goal|target|objective|"\n'
    '        r"budget|timeline|schedule|roadmap|strategy|kpi|okr)\\b"\n'
    '        r"|\\bdedicated\\s+to\\s+(?:the\\s+)?(?:gym|fitness|working\\s+out|diet|exercise|routine)\\b",\n'
    '        re.IGNORECASE,\n'
    '    ),'
)

assert src.count(old_commitment) == 1, f"Expected 1 commitment match, got {src.count(old_commitment)}"
src = src.replace(old_commitment, new_commitment)

with open('C:/Ethos/core/value_extractor.py', 'w', encoding='utf-8') as f:
    f.write(src)

# Verify
with open('C:/Ethos/core/value_extractor.py', encoding='utf-8') as f:
    result = f.read()

assert '"growth"' in result and 'medical' in result, "growth disqualifier missing"
assert '"independence"' in result and 'trivial' in result, "independence disqualifier missing"
assert 'kpi' in result, "commitment expansion missing"
print("All 3 gap-closing changes applied.")
