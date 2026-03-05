"""Fix independence and commitment disqualifier patterns."""
import re

with open('C:/Ethos/core/value_extractor.py', encoding='utf-8') as f:
    src = f.read()

# Verify current independence pattern is there
assert '"independence"' in src, "independence entry missing"
assert '"commitment"' in src, "commitment entry missing"

# Replace independence disqualifier — add activity-BEFORE-keyword pattern
# Find by locating the unique comment line above it
old_indep_comment = '    # "on my own to the store" / "by myself shopping" -- trivial, not independence-as-value'
new_indep_block = r'''    # "on my own to the store" / "cooking by myself" -- trivial self-sufficiency, not the value
    "independence": re.compile(
        r"\b(?:on\s+my\s+own|by\s+myself)\s+"
        r"(?:to\s+(?:the|a)\s+\w+|at\s+(?:the|a)\s+\w+|"
        r"shopping|cooking|driving|walking|cleaning|playing|running|working\s+out)\b"
        r"|\b(?:shopping|cooking|driving|walking|cleaning|playing|running|working\s+out)"
        r"\s+(?:on\s+my\s+own|by\s+myself)\b",
        re.IGNORECASE,
    ),'''

# Find and replace the old independence block
indep_start = src.index(old_indep_comment)
# Find the end of this entry (next entry starts with 4 spaces + ")" + ",")
# The entry ends at    ),\n    # next comment
indep_block_end = src.index('\n    # "on my own to the store"', indep_start - 1)
# Actually: find from indep_start to the closing ),
after_indep = src[indep_start:]
# Find the closing ),\n
close_idx = after_indep.index('\n    ),\n') + len('\n    ),\n')
old_indep_block = after_indep[:close_idx]
src = src.replace(old_indep_block, '\n' + new_indep_block + '\n', 1)

# Expand commitment disqualifier — add corp-speak verbs
# Current pattern ends with: r"budget|timeline|schedule|roadmap|strategy|kpi|okr)\b",
# We add another alternation line
old_commit_end = '        r"|\\bcommitted\\s+to\\s+(?:the\\s+)?(?:project|plan|goal|target|objective|"\n        r"budget|timeline|schedule|roadmap|strategy|kpi|okr)\\b"\n        r"|\\bdedicated\\s+to'
new_commit_end = '        r"|\\bcommitted\\s+to\\s+(?:the\\s+)?(?:project|plan|goal|target|objective|"\n        r"budget|timeline|schedule|roadmap|strategy|kpi|okr)\\b"\n        r"|\\bcommitted\\s+to\\s+(?:achieving|reaching|hitting|meeting|delivering|"\n        r"executing|fulfilling|implementing|completing)\\b"\n        r"|\\bdedicated\\s+to'

assert src.count(old_commit_end) == 1, f"Expected 1 commitment match, got {src.count(old_commit_end)}"
src = src.replace(old_commit_end, new_commit_end)

with open('C:/Ethos/core/value_extractor.py', 'w', encoding='utf-8') as f:
    f.write(src)

# Quick syntax check
import py_compile
try:
    py_compile.compile('C:/Ethos/core/value_extractor.py', doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
    raise

print("All pattern fixes applied.")
