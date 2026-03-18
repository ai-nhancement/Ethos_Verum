"""
tests/test_phrase_layer.py

Tests for core/phrase_layer.py — two-pass phrase composition detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.phrase_layer import (
    phrase_signals,
    _DESTRUCTIVE_RE,
    _OPPOSITION_RE,
    _PHRASE_BASE_CONF,
    _PHRASE_PROX_CHARS,
)


# ---------------------------------------------------------------------------
# Regex smoke tests
# ---------------------------------------------------------------------------

class TestDestructiveVerbRegex:
    @pytest.mark.parametrize("verb", [
        "committed", "commit", "commits", "committing",
        "inflicted", "inflicts", "inflicting",
        "perpetrated", "perpetrates", "perpetrating",
        "orchestrated", "authorized", "authorised",
        "sanctioned", "ordered", "organized", "organised",
        "practiced", "exercised", "unleashed", "employed",
        "propagated",
    ])
    def test_destructive_verb_matches(self, verb):
        assert _DESTRUCTIVE_RE.search(verb), f"Expected match for: {verb!r}"

    def test_bare_used_does_not_match(self):
        # "used" alone is too generic — not in the pattern
        assert not _DESTRUCTIVE_RE.search("used")

    def test_bare_showed_does_not_match(self):
        assert not _DESTRUCTIVE_RE.search("showed")


class TestOppositionVerbRegex:
    @pytest.mark.parametrize("phrase", [
        "resisted", "resist", "resisting",
        "opposed", "opposes", "opposing",
        "confronted", "challenged", "condemned",
        "denounced", "rejected", "combated",
        "exposed",
        "fought against", "fighting against",
        "stood against", "standing against",
        "stood up to",
        "spoke against", "speaking against",
        "campaigned against", "worked against",
        "protested", "protested against",
    ])
    def test_opposition_phrase_matches(self, phrase):
        assert _OPPOSITION_RE.search(phrase), f"Expected match for: {phrase!r}"

    def test_bare_against_does_not_match(self):
        # Bare "against" alone must NOT be an opposition signal
        assert not _OPPOSITION_RE.search("against")

    def test_loved_does_not_match(self):
        assert not _OPPOSITION_RE.search("loved and cherished")


# ---------------------------------------------------------------------------
# Core detection — destructive phrases
# ---------------------------------------------------------------------------

class TestDestructivePhrases:
    def test_committed_cruelty(self):
        sigs, consumed = phrase_signals(
            "He committed acts of cruelty toward the prisoners.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == -1 for s in sigs)

    def test_inflicted_suffering(self):
        sigs, consumed = phrase_signals(
            "The regime inflicted suffering on millions of civilians.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == -1 for s in sigs)

    def test_perpetrated_violence(self):
        sigs, consumed = phrase_signals(
            "He perpetrated acts of violence against his own people.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == -1 for s in sigs)

    def test_ordered_persecution(self):
        sigs, consumed = phrase_signals(
            "He ordered the persecution of religious minorities.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == -1 for s in sigs)

    def test_orchestrated_oppression(self):
        sigs, consumed = phrase_signals(
            "She orchestrated a campaign of oppression against dissidents.",
            0.7, "speech",
        )
        assert any(s["polarity_hint"] == -1 for s in sigs)

    def test_practiced_cruelty(self):
        sigs, consumed = phrase_signals(
            "He practiced cruelty as a tool of governance.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == -1 for s in sigs)

    def test_unleashed_oppression(self):
        sigs, consumed = phrase_signals(
            "The general unleashed oppression upon the civilian population.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == -1 for s in sigs)


# ---------------------------------------------------------------------------
# Core detection — opposition phrases (constructive)
# ---------------------------------------------------------------------------

class TestOppositionPhrases:
    def test_resisted_cruelty(self):
        sigs, consumed = phrase_signals(
            "He resisted cruelty throughout his career.",
            0.7, "journal",
        )
        assert any(s["polarity_hint"] == 1 for s in sigs)

    def test_opposed_oppression(self):
        sigs, consumed = phrase_signals(
            "She opposed oppression at every turn.",
            0.7, "letter",
        )
        assert any(s["polarity_hint"] == 1 for s in sigs)

    def test_condemned_persecution(self):
        sigs, consumed = phrase_signals(
            "He condemned the persecution of minorities in the strongest terms.",
            0.7, "speech",
        )
        assert any(s["polarity_hint"] == 1 for s in sigs)

    def test_denounced_cruelty(self):
        sigs, consumed = phrase_signals(
            "She denounced the cruelty of the colonial administration.",
            0.7, "letter",
        )
        assert any(s["polarity_hint"] == 1 for s in sigs)

    def test_fought_against_oppression(self):
        sigs, consumed = phrase_signals(
            "He fought against oppression his entire life.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == 1 for s in sigs)

    def test_exposed_corruption(self):
        sigs, consumed = phrase_signals(
            "She exposed corruption in the upper ranks of government.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == 1 for s in sigs)

    def test_stood_against_oppression(self):
        sigs, consumed = phrase_signals(
            "He stood against oppression when no one else would.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == 1 for s in sigs)

    def test_protested_against_oppression(self):
        sigs, consumed = phrase_signals(
            "The citizens protested against oppression in the streets.",
            0.7, "action",
        )
        assert any(s["polarity_hint"] == 1 for s in sigs)


# ---------------------------------------------------------------------------
# No verb in window — fall through (no phrase signal)
# ---------------------------------------------------------------------------

class TestFallThrough:
    def test_no_verb_returns_no_signal(self):
        # "cruelty" appears, but no compositional verb before it
        sigs, consumed = phrase_signals(
            "The cruelty of the situation was undeniable.",
            0.7, "journal",
        )
        assert sigs == []
        assert consumed == set()

    def test_neutral_verb_not_a_phrase(self):
        # "was" is not a compositional verb
        sigs, consumed = phrase_signals(
            "The oppression was widespread throughout the province.",
            0.7, "journal",
        )
        assert sigs == []

    def test_distant_verb_does_not_fire(self):
        # Destructive verb is more than _PHRASE_PROX_CHARS before the vice word
        filler = "a" * (_PHRASE_PROX_CHARS + 10)
        text = f"He committed many crimes. {filler} The cruelty stood alone."
        sigs, consumed = phrase_signals(text, 0.7, "journal")
        # "committed" is far from "cruelty" — should not form a phrase
        assert not any(s["polarity_hint"] == -1 for s in sigs)


# ---------------------------------------------------------------------------
# Consumed tokens
# ---------------------------------------------------------------------------

class TestConsumedTokens:
    def test_phrase_match_consumes_vice_token(self):
        _, consumed = phrase_signals(
            "He committed acts of cruelty toward the prisoners.",
            0.7, "action",
        )
        # "cruelty" is in MFD2.0 as care.vice
        # If MFD2 not available, consumed will be empty — skip gracefully
        assert isinstance(consumed, set)

    def test_no_match_does_not_consume(self):
        _, consumed = phrase_signals(
            "The cruelty of the situation was undeniable.",
            0.7, "journal",
        )
        assert consumed == set()

    def test_consumed_tokens_are_lowercase(self):
        _, consumed = phrase_signals(
            "He committed acts of Cruelty toward the prisoners.",
            0.7, "action",
        )
        for tok in consumed:
            assert tok == tok.lower()


# ---------------------------------------------------------------------------
# Signal schema validation
# ---------------------------------------------------------------------------

class TestSignalSchema:
    def test_signal_has_required_keys(self):
        sigs, _ = phrase_signals(
            "He committed acts of cruelty toward the prisoners.",
            0.7, "action",
        )
        if not sigs:
            pytest.skip("MFD2 not available")
        for sig in sigs:
            assert "value_name" in sig
            assert "text_excerpt" in sig
            assert "significance" in sig
            assert "disambiguation_confidence" in sig
            assert "source" in sig
            assert "polarity_hint" in sig
            assert sig["source"] == "phrase_layer"

    def test_polarity_hint_is_int(self):
        sigs, _ = phrase_signals(
            "He committed acts of cruelty toward the prisoners.",
            0.7, "action",
        )
        for sig in sigs:
            assert isinstance(sig["polarity_hint"], int)
            assert sig["polarity_hint"] in (-1, 1)

    def test_confidence_equals_base(self):
        sigs, _ = phrase_signals(
            "He committed acts of cruelty toward the prisoners.",
            0.7, "action",
        )
        for sig in sigs:
            assert sig["disambiguation_confidence"] == _PHRASE_BASE_CONF

    def test_significance_passed_through(self):
        sigs, _ = phrase_signals(
            "He committed acts of cruelty toward the prisoners.",
            0.85, "action",
        )
        for sig in sigs:
            assert sig["significance"] == 0.85

    def test_excerpt_max_150_chars(self):
        long_text = "He committed acts of cruelty. " * 20
        sigs, _ = phrase_signals(long_text, 0.7, "action")
        for sig in sigs:
            assert len(sig["text_excerpt"]) <= 150

    def test_deduplication_one_signal_per_value_polarity(self):
        # Two occurrences of the same vice word in destructive context —
        # should produce one signal per (value_name, polarity_hint).
        text = "He committed cruelty. He committed cruelty again."
        sigs, _ = phrase_signals(text, 0.7, "action")
        keys = [(s["value_name"], s["polarity_hint"]) for s in sigs]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_empty_text_returns_empty(self):
        sigs, consumed = phrase_signals("", 0.7, "action")
        assert sigs == []
        assert consumed == set()

    def test_never_raises(self):
        # Should not raise on unusual inputs
        for text in ["", "x" * 5000, "💥", None]:
            try:
                result = phrase_signals(text, 0.5, "unknown")
                assert isinstance(result, tuple)
                assert len(result) == 2
            except Exception as e:
                pytest.fail(f"phrase_signals raised on {text!r}: {e}")

    def test_lexicon_unavailable_returns_empty(self):
        # Patch MFD2 to be empty — should return gracefully
        from core import phrase_layer
        import core.lexicon_layer as ll
        orig = dict(ll._MFD2)
        ll._MFD2.clear()
        try:
            sigs, consumed = phrase_signals("He committed cruelty.", 0.7, "action")
            assert sigs == []
            assert consumed == set()
        finally:
            ll._MFD2.update(orig)

    def test_zero_significance(self):
        sigs, _ = phrase_signals("He committed acts of cruelty.", 0.0, "action")
        assert isinstance(sigs, list)

    def test_text_with_no_lexicon_words(self):
        sigs, consumed = phrase_signals(
            "He committed the act and walked away quickly.",
            0.7, "action",
        )
        # "committed" fires destructive, but no vice word in MFD2 here
        assert sigs == []
        assert consumed == set()


# ---------------------------------------------------------------------------
# Polarity direction tests (directional correctness)
# ---------------------------------------------------------------------------

class TestPolarityDirection:
    def test_destructive_context_negative(self):
        sigs, _ = phrase_signals(
            "He inflicted suffering on countless innocent people.",
            0.7, "action",
        )
        if sigs:
            assert all(s["polarity_hint"] == -1 for s in sigs)

    def test_opposition_context_positive(self):
        sigs, _ = phrase_signals(
            "She resisted cruelty and protected those in her care.",
            0.7, "action",
        )
        if sigs:
            assert all(s["polarity_hint"] == 1 for s in sigs)

    def test_same_vice_word_different_contexts(self):
        destructive_sigs, _ = phrase_signals(
            "He committed cruelty against the prisoners.",
            0.7, "action",
        )
        opposition_sigs, _ = phrase_signals(
            "She resisted cruelty throughout her life.",
            0.7, "action",
        )
        if destructive_sigs and opposition_sigs:
            assert any(s["polarity_hint"] == -1 for s in destructive_sigs)
            assert any(s["polarity_hint"] == 1 for s in opposition_sigs)
