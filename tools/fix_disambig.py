"""Fix 2 bugs in value_extractor.py post-transformation."""

with open('C:/Ethos/core/value_extractor.py', encoding='utf-8') as f:
    src = f.read()

# ── FIX 1: love disqualifier — handle "loves" (plural verb form) ─────────────
src = src.replace(
    r'r"\b(?:I\s+)?love\s+(?:pizza|coffee|tea|beer|wine|food|sports?|movies?|music|games?|this|that|it|them)\b",',
    r'r"\b(?:I\s+)?loves?\s+(?:pizza|coffee|tea|beer|wine|food|sports?|movies?|music|games?|this|that|it|them)\b",'
)

# ── FIX 2: Overlap-based disqualifier matching ────────────────────────────────
# Replace context-window disqualifier check with keyword-overlap check so that
# "brave face" in sentence doesn't block "courageous" in same sentence.
old_disq_check = '''\
        # Check 1: disqualifier
        disq = _DISQUALIFIERS.get(value_name)
        if disq and disq.search(ctx):
            return False, 0.0'''

new_disq_check = '''\
        # Check 1: disqualifier -- overlap-based: only block if the disqualifier
        # match overlaps with the actual keyword position (not just nearby text).
        disq = _DISQUALIFIERS.get(value_name)
        if disq:
            kw_end = match_idx + len(kw)
            for m in disq.finditer(text_lower):
                if m.start() < kw_end and m.end() > match_idx:
                    return False, 0.0'''

src = src.replace(old_disq_check, new_disq_check)

with open('C:/Ethos/core/value_extractor.py', 'w', encoding='utf-8') as f:
    f.write(src)

# Verify both changes landed
with open('C:/Ethos/core/value_extractor.py', encoding='utf-8') as f:
    result = f.read()

assert 'loves?' in result, "love plural fix missing"
assert 'overlap-based' in result, "overlap-based disqualifier fix missing"
print("Both fixes applied.")
