"""
core/comprehension_panel.py

Three-model comprehension panel for value signal verification.

Uses three strong models via the DigitalOcean Gradient API to independently
verify whether detected value signals are genuine — via two binary questions
per value:

  Q1: Does this passage show the figure HOLDING this value?  → P1 signal
  Q2: Does this passage show the figure VIOLATING this value? → P0 signal

Majority vote (≥ 2 of 3) determines the final verdict per value:
  "P1"      — ≥2 Q1=yes, Q2=no
  "P0"      — ≥2 Q2=yes, Q1=no
  "tension" — ≥2 both Q1 and Q2 yes
  "discard" — ≥2 neither yes
  "skip"    — panel unavailable, 2+ model failures, or config disabled

Panel calls are fail-open: a failed model counts as abstain.
If 2+ models fail (abstain), the verdict falls through to "skip" and the
original signal passes through unchanged.

Required:
  pip install gradient
  MODEL_ACCESS_KEY env var (DigitalOcean Gradient API key)

Models used (configurable via _MODEL_IDS below):
  anthropic-claude-opus-4.6   — Anthropic Claude Opus 4.6  ($5/$25 per 1M)
  openai-gpt-5.4              — OpenAI GPT-5.4             ($2.50/$15 per 1M)
  openai-o3                   — OpenAI o3 (reasoning)      ($2/$8 per 1M)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifiers — DigitalOcean Gradient model names
#
# PREFERRED (unlock by upgrading account to premium tier):
#   "anthropic-claude-opus-4.6"          # Claude Opus 4.6    $5/$25 per 1M
#   "openai-gpt-5.4"                     # GPT-5.4            $2.50/$15 per 1M
#   "openai-o3"                          # o3 reasoning       $2/$8 per 1M
#
# CURRENT (available on base tier):
# ---------------------------------------------------------------------------

_MODEL_IDS = [
    "openai-gpt-oss-120b",              # 120B OSS model     $0.10/$0.70 per 1M
    "deepseek-r1-distill-llama-70b",    # DeepSeek R1 70B    $0.99/$0.99 per 1M
    "openai-gpt-oss-20b",              # 20B OSS model      $0.05/$0.45 per 1M
]

# Timeout per model call (seconds)
_CALL_TIMEOUT = 45

# Max output tokens per panel call (JSON with up to ~15 values)
_MAX_TOKENS = 1024

# Env var name for the DigitalOcean Gradient API key
_KEY_ENV_VAR = "MODEL_ACCESS_KEY"

# ---------------------------------------------------------------------------
# Verdict type
# ---------------------------------------------------------------------------

Verdict = str  # "P1" | "P0" | "tension" | "discard" | "skip"

# Per-value result from a single model: (holds, violates)
_ModelBinary = Tuple[Optional[bool], Optional[bool]]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise historical analysis assistant. "
    "Your task is to determine whether a biographical passage provides "
    "evidence about a specific value for the named figure. "
    "Respond ONLY with valid JSON — no explanation, no markdown."
)

_USER_TEMPLATE = """\
PASSAGE:
{text}

FIGURE: {figure_name}

For each value listed below, assess whether this passage provides clear \
textual evidence that the figure HOLDS the value (demonstrates, upholds, or \
acts in accordance with it) or VIOLATES the value (betrays, acts against, or \
fails to uphold it).

Values: {value_list}

Respond with a JSON object. Each key is a value name. Each value is an object \
with exactly two boolean fields:
  "holds"    — true if the passage shows the figure holding/demonstrating this value
  "violates" — true if the passage shows the figure violating/betraying this value

Only mark true when there is direct, unambiguous evidence in the passage. \
Ambiguity or absence of evidence → false.

Example format:
{{"integrity": {{"holds": true, "violates": false}}, \
"courage": {{"holds": false, "violates": false}}}}
"""


def _build_user_prompt(text: str, figure_name: str, value_names: List[str]) -> str:
    excerpt = text[:1500]
    fn = figure_name if figure_name else "the figure"
    return _USER_TEMPLATE.format(
        text=excerpt,
        figure_name=fn,
        value_list=", ".join(value_names),
    )


# ---------------------------------------------------------------------------
# Single model call (sync, runs in thread pool for parallelism)
# ---------------------------------------------------------------------------

def _call_model(model_id: str, user_prompt: str) -> Optional[Dict[str, _ModelBinary]]:
    """
    Call one Gradient model synchronously.
    Returns parsed result dict or None on any failure.
    """
    try:
        from gradient import Gradient  # type: ignore
    except ImportError:
        return None

    api_key = os.environ.get(_KEY_ENV_VAR, "")
    if not api_key:
        _load_dotenv()
        api_key = os.environ.get(_KEY_ENV_VAR, "")
    if not api_key:
        return None

    try:
        client = Gradient(model_access_key=api_key)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            model=model_id,
            max_tokens=_MAX_TOKENS,
        )
        text = response.choices[0].message.content or ""
        return _parse_model_response(text)
    except Exception as exc:
        _log.debug("comprehension_panel: %s call failed: %s", model_id, exc)
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_model_response(
    text: str,
) -> Optional[Dict[str, _ModelBinary]]:
    """
    Parse a model's JSON response into {value_name: (holds, violates)} dict.
    Returns None if the response cannot be parsed.
    """
    text = text.strip()
    if not text:
        return None

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    result: Dict[str, _ModelBinary] = {}
    for value_name, judgment in data.items():
        if not isinstance(judgment, dict):
            continue
        holds    = judgment.get("holds")
        violates = judgment.get("violates")
        holds    = bool(holds)    if isinstance(holds, bool)    else None
        violates = bool(violates) if isinstance(violates, bool) else None
        result[str(value_name).lower()] = (holds, violates)

    return result if result else None


# ---------------------------------------------------------------------------
# Vote tallying
# ---------------------------------------------------------------------------

def _tally_votes(
    responses: List[Optional[Dict[str, _ModelBinary]]],
    value_name: str,
) -> Verdict:
    """
    Given up to 3 model responses, compute the majority-vote verdict for
    one value.

    Abstain (None response or missing key) counts as neither yes nor no.
    If 2+ models abstained → "skip" (panel unavailable for this value).
    """
    holds_yes    = 0
    violates_yes = 0
    abstain      = 0

    for resp in responses:
        if resp is None:
            abstain += 1
            continue
        pair = resp.get(value_name)
        if pair is None:
            abstain += 1
            continue
        holds, violates = pair
        if holds    is True:
            holds_yes    += 1
        if violates is True:
            violates_yes += 1

    active = len(responses) - abstain
    if active < 2:
        return "skip"

    majority = 2  # ≥ 2 of 3

    both_held     = holds_yes    >= majority
    both_violated = violates_yes >= majority

    if both_held and both_violated:
        return "tension"
    if both_held:
        return "P1"
    if both_violated:
        return "P0"
    return "discard"


# ---------------------------------------------------------------------------
# Core async panel runner
# ---------------------------------------------------------------------------

async def _run_panel_async(
    text: str,
    figure_name: str,
    value_names: List[str],
) -> Dict[str, Verdict]:
    """
    Call all three models in parallel (via thread pool — Gradient SDK is sync)
    and return per-value verdicts.
    """
    user_prompt = _build_user_prompt(text, figure_name, value_names)

    tasks = [
        asyncio.to_thread(_call_model, model_id, user_prompt)
        for model_id in _MODEL_IDS
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=False)

    verdicts: Dict[str, Verdict] = {}
    for vname in value_names:
        verdicts[vname] = _tally_votes(list(responses), vname)

    return verdicts


# ---------------------------------------------------------------------------
# Public sync entry point
# ---------------------------------------------------------------------------

def verify_signals(
    text: str,
    figure_name: str,
    signals: List[Dict],
    enabled: bool = True,
) -> List[Dict]:
    """
    Run the three-model comprehension panel on a set of candidate signals for
    a single passage. Returns a filtered/updated list of signals.

    Signals with verdict "skip"    — pass through unchanged
    Signals with verdict "discard" — removed
    Signals with verdict "P1"      — polarity_hint set to +1
    Signals with verdict "P0"      — polarity_hint set to -1
    Signals with verdict "tension" — duplicated as both P1 and P0 entries

    Always returns a list; never raises.
    """
    if not enabled or not signals:
        return signals

    value_names = list({s["value_name"] for s in signals})
    if not value_names:
        return signals

    try:
        verdicts = _run_async(
            _run_panel_async(text, figure_name, value_names)
        )
    except Exception as exc:
        _log.debug("comprehension_panel: panel run failed: %s", exc)
        return signals  # fail-open

    updated: List[Dict] = []
    for sig in signals:
        vname   = sig["value_name"]
        verdict = verdicts.get(vname, "skip")

        if verdict == "skip":
            updated.append(sig)

        elif verdict == "discard":
            _log.debug(
                "comprehension_panel: discarded signal value=%s  panel=discard",
                vname,
            )

        elif verdict == "P1":
            s = dict(sig)
            s["polarity_hint"] = 1
            s["source"] = s.get("source", "") + "+panel"
            updated.append(s)

        elif verdict == "P0":
            s = dict(sig)
            s["polarity_hint"] = -1
            s["source"] = s.get("source", "") + "+panel"
            updated.append(s)

        elif verdict == "tension":
            p1 = dict(sig)
            p1["polarity_hint"] = 1
            p1["source"] = p1.get("source", "") + "+panel"
            p0 = dict(sig)
            p0["polarity_hint"] = -1
            p0["source"] = p0.get("source", "") + "+panel"
            updated.extend([p1, p0])
            _log.debug(
                "comprehension_panel: tension detected value=%s — emitting both polarities",
                vname,
            )

        else:
            updated.append(sig)

    return updated


def _run_async(coro):
    """
    Run an async coroutine from synchronous code.
    Uses a thread pool if a running event loop exists (e.g., inside FastAPI).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# .env loader (optional convenience — falls back silently if not installed)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """
    Returns True if the gradient SDK is importable and MODEL_ACCESS_KEY is set.
    """
    try:
        import gradient  # noqa: F401
    except ImportError:
        return False

    _load_dotenv()
    return bool(os.environ.get(_KEY_ENV_VAR, ""))
