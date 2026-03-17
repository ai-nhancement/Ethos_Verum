"""
tests/test_structural_layer.py

Dedicated tests for core/structural_layer.py — the Layer 3 structural
pattern detector and zero-shot entailment sub-layer.

These are isolated unit tests. They do not require the full pipeline,
any database, or the DeBERTa model (zero-shot tests are mocked unless
the model is available).

Test structure:
  TestTierConstants          — _T1, _T2, _T3 values and ordering
  TestAdversityTiers         — _ADV_T1 / _ADV_T2 / _ADV_T3 regex discrimination
  TestAgencyTiers            — _AGC_T1 / _AGC_T2 / _AGC_T3 regex discrimination
  TestResistanceTiers        — _RES_T1 / _RES_T2 / _RES_T3 regex discrimination
  TestStakesTiers            — _STK_T1 / _STK_T2 / _STK_T3 regex discrimination
  TestClassScore             — _class_score() helper
  TestStructuralScore        — structural_score() continuous range and calibration
  TestTierDiscrimination     — key invariant: higher-intensity language → higher score
  TestZeroshotScores         — zeroshot_scores() with mocked pipeline
  TestLayer3Signals          — layer3_signals() integration
  TestRegexEdgeCases         — boundary conditions and non-matching text
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.structural_layer import (
    _T1, _T2, _T3,
    _ADV_T1, _ADV_T2, _ADV_T3,
    _AGC_T1, _AGC_T2, _AGC_T3,
    _RES_T1, _RES_T2, _RES_T3,
    _STK_T1, _STK_T2, _STK_T3,
    _class_score,
    structural_score,
    layer3_signals,
    zeroshot_scores,
    VALUE_HYPOTHESES,
)


# ---------------------------------------------------------------------------
# TestTierConstants
# ---------------------------------------------------------------------------

class TestTierConstants:
    def test_t1_is_lowest(self):
        assert _T1 < _T2 < _T3

    def test_t3_is_one(self):
        assert _T3 == 1.0

    def test_t1_positive(self):
        assert _T1 > 0.0

    def test_t2_between_t1_and_t3(self):
        assert _T1 < _T2 < _T3


# ---------------------------------------------------------------------------
# TestAdversityTiers
# ---------------------------------------------------------------------------

class TestAdversityTiers:
    """
    T1 (concessive) < T2 (real cost/pressure) < T3 (existential/irreversible).

    Critical invariant: a T1 phrase must NOT match T2 or T3.
    A T2 phrase must NOT match T3.
    """

    # --- T1 matches ---
    def test_t1_despite(self):
        assert _ADV_T1.search("Despite the difficulty he continued.")

    def test_t1_in_spite_of(self):
        assert _ADV_T1.search("In spite of the opposition, she held firm.")

    def test_t1_although(self):
        assert _ADV_T1.search("Although they threatened him, he refused.")

    def test_t1_even_though(self):
        assert _ADV_T1.search("Even though it cost him everything, he told the truth.")

    def test_t1_notwithstanding(self):
        assert _ADV_T1.search("Notwithstanding the risks, she pressed on.")

    # --- T1 phrases do NOT match T2 or T3 ---
    def test_despite_not_t2(self):
        assert not _ADV_T2.search("Despite the difficulty.")

    def test_despite_not_t3(self):
        assert not _ADV_T3.search("Despite the difficulty.")

    def test_although_not_t2(self):
        assert not _ADV_T2.search("Although he was nervous.")

    # --- T2 matches ---
    def test_t2_at_great_risk(self):
        assert _ADV_T2.search("He spoke at great risk to his own safety.")

    def test_t2_at_personal_cost(self):
        assert _ADV_T2.search("She acted at personal cost to herself.")

    def test_t2_knowing_the_cost(self):
        assert _ADV_T2.search("Knowing the cost, he signed the declaration.")

    def test_t2_knowing_the_risk(self):
        assert _ADV_T2.search("Knowing the risk, she stepped forward.")

    def test_t2_knowing_the_consequences(self):
        assert _ADV_T2.search("Knowing the consequences, he spoke anyway.")

    def test_t2_under_pressure(self):
        assert _ADV_T2.search("He maintained his position under pressure.")

    def test_t2_under_threat(self):
        assert _ADV_T2.search("She refused to yield under threat.")

    def test_t2_under_interrogation(self):
        assert _ADV_T2.search("He would not confess even under interrogation.")

    def test_t2_against_all_opposition(self):
        assert _ADV_T2.search("She pressed forward against all opposition.")

    def test_t2_in_the_face_of(self):
        assert _ADV_T2.search("In the face of death, he did not waver.")

    def test_t2_at_the_expense_of(self):
        assert _ADV_T2.search("He told the truth at the expense of his career.")

    def test_t2_against_all_odds(self):
        assert _ADV_T2.search("She succeeded against the odds.")

    # --- T2 phrases do NOT match T3 ---
    def test_under_pressure_not_t3(self):
        assert not _ADV_T3.search("He stood firm under pressure.")

    def test_at_great_risk_not_t3(self):
        assert not _ADV_T3.search("She spoke at great risk.")

    # --- T3 matches ---
    def test_t3_risking_everything(self):
        assert _ADV_T3.search("Risking everything, she exposed the conspiracy.")

    def test_t3_risking_my_life(self):
        assert _ADV_T3.search("Risking my life, I carried the message.")

    def test_t3_risking_his_life(self):
        assert _ADV_T3.search("He crossed the border risking his life.")

    def test_t3_sacrificing_everything(self):
        assert _ADV_T3.search("Sacrificing everything, she spoke out.")

    def test_t3_sacrificed_career(self):
        assert _ADV_T3.search("He sacrificed his career to expose the corruption.")

    def test_t3_gave_up_everything(self):
        assert _ADV_T3.search("She gave up everything for the cause.")

    def test_t3_laid_down_his_life(self):
        assert _ADV_T3.search("He laid down his life for his comrades.")

    def test_t3_laid_down_her_life(self):
        assert _ADV_T3.search("She laid down her life protecting the children.")

    # --- Neutral text matches nothing ---
    def test_neutral_matches_no_tier(self):
        text = "The meeting was scheduled for Tuesday morning."
        assert not _ADV_T1.search(text)
        assert not _ADV_T2.search(text)
        assert not _ADV_T3.search(text)


# ---------------------------------------------------------------------------
# TestAgencyTiers
# ---------------------------------------------------------------------------

class TestAgencyTiers:
    """
    T1: passive continuation verbs (continued, remained, stayed).
    T2: active resistance verbs (refused, chose, stood, held, persisted).
    T3: explicit named refusal of a failure act (would not, did not, could not betray).

    All require first-person subject (I|we).
    """

    # --- T1 matches ---
    def test_t1_i_continued(self):
        assert _AGC_T1.search("I continued despite the exhaustion.")

    def test_t1_i_remained(self):
        assert _AGC_T1.search("I remained at my post throughout the siege.")

    def test_t1_i_stayed(self):
        assert _AGC_T1.search("I stayed even when the others left.")

    def test_t1_i_kept_going(self):
        assert _AGC_T1.search("I kept going after every setback.")

    def test_t1_i_yet_continued(self):
        assert _AGC_T1.search("I yet continued even after the losses.")

    def test_t1_i_still_stayed(self):
        assert _AGC_T1.search("I still stayed when the others fled.")

    # --- T1 verbs do NOT match T2 or T3 ---
    def test_i_continued_not_t2(self):
        assert not _AGC_T2.search("I continued despite the danger.")

    def test_i_remained_not_t3(self):
        assert not _AGC_T3.search("I remained at my post.")

    # --- T2 matches ---
    def test_t2_i_refused(self):
        assert _AGC_T2.search("I refused to abandon my post.")

    def test_t2_i_refuse(self):
        assert _AGC_T2.search("I refuse to betray my principles.")

    def test_t2_i_stood(self):
        assert _AGC_T2.search("I stood despite the overwhelming pressure.")

    def test_t2_i_chose(self):
        assert _AGC_T2.search("I chose to speak even knowing the cost.")

    def test_t2_i_held(self):
        assert _AGC_T2.search("I held my ground when they demanded I step aside.")

    def test_t2_i_pressed_on(self):
        assert _AGC_T2.search("I pressed on despite exhaustion.")

    def test_t2_i_still_refused(self):
        assert _AGC_T2.search("I still refused to yield.")

    def test_t2_i_nevertheless_persisted(self):
        assert _AGC_T2.search("I nevertheless persisted through the ordeal.")

    def test_t2_we_endured(self):
        assert _AGC_T2.search("We endured every hardship without complaint.")

    def test_t2_i_rose(self):
        assert _AGC_T2.search("I rose after each defeat.")

    # --- T2 phrases do NOT match T3 ---
    def test_i_refused_not_t3(self):
        # "refused" alone (without "did not" / "would not") is T2, not T3
        assert not _AGC_T3.search("I refused to surrender.")

    # --- T3 matches ---
    def test_t3_i_would_not(self):
        assert _AGC_T3.search("I would not abandon them.")

    def test_t3_i_did_not(self):
        assert _AGC_T3.search("I did not yield to their demands.")

    def test_t3_i_could_not_abandon(self):
        assert _AGC_T3.search("I could not abandon the people who trusted me.")

    def test_t3_i_could_not_betray(self):
        assert _AGC_T3.search("I could not betray those who depended on me.")

    def test_t3_we_would_not(self):
        assert _AGC_T3.search("We would not deceive them even to save ourselves.")

    # --- Third-person does NOT match any tier ---
    def test_third_person_not_t1(self):
        assert not _AGC_T1.search("He continued despite the danger.")

    def test_third_person_not_t2(self):
        assert not _AGC_T2.search("She refused to yield.")

    def test_third_person_not_t3(self):
        assert not _AGC_T3.search("He would not betray his men.")

    # --- Regression: 'chose' not 'chos' ---
    def test_chose_matches(self):
        assert _AGC_T2.search("I chose to remain.")

    def test_chos_does_not_match(self):
        assert not _AGC_T2.search("I chos the harder path.")


# ---------------------------------------------------------------------------
# TestResistanceTiers
# ---------------------------------------------------------------------------

class TestResistanceTiers:
    """
    T1: did-not / never constructions for positional failure.
    T2: explicit refused/would-not for tactical/positional failure.
    T3: refused to betray/abandon/deceive — moral and relational stakes.
    """

    # --- T1 matches ---
    def test_t1_did_not_waver(self):
        assert _RES_T1.search("She did not waver throughout the ordeal.")

    def test_t1_did_not_falter(self):
        assert _RES_T1.search("He did not falter under the pressure.")

    def test_t1_did_not_yield(self):
        assert _RES_T1.search("She did not yield despite the threats.")

    def test_t1_did_not_retreat(self):
        assert _RES_T1.search("He did not retreat from his position.")

    def test_t1_never_yielded(self):
        assert _RES_T1.search("She never yielded on the matter of principle.")

    def test_t1_never_faltered(self):
        assert _RES_T1.search("He never faltered in his resolve.")

    # --- T1 phrases do NOT match T2 or T3 ---
    def test_did_not_waver_not_t2(self):
        assert not _RES_T2.search("She did not waver.")

    def test_did_not_yield_not_t3(self):
        assert not _RES_T3.search("He did not yield.")

    # --- T2 matches ---
    def test_t2_refused_to_yield(self):
        assert _RES_T2.search("He refused to yield even under torture.")

    def test_t2_refused_to_surrender(self):
        assert _RES_T2.search("She refused to surrender her principles.")

    def test_t2_refused_to_flee(self):
        assert _RES_T2.search("She refused to flee when others ran.")

    def test_t2_refused_to_give_up(self):
        assert _RES_T2.search("I refused to give up despite everything.")

    def test_t2_refused_to_back_down(self):
        assert _RES_T2.search("He refused to back down from his position.")

    def test_t2_would_not_yield(self):
        assert _RES_T2.search("She would not yield to their demands.")

    def test_t2_would_not_be_silenced(self):
        assert _RES_T2.search("He would not be silenced by the authorities.")

    # --- T2 phrases do NOT match T3 ---
    def test_refused_to_flee_not_t3(self):
        assert not _RES_T3.search("She refused to flee.")

    def test_refused_to_yield_not_t3(self):
        assert not _RES_T3.search("He refused to yield.")

    # --- T3 matches ---
    def test_t3_refused_to_betray(self):
        assert _RES_T3.search("He refused to betray his comrades.")

    def test_t3_refused_to_abandon(self):
        assert _RES_T3.search("She refused to abandon the injured.")

    def test_t3_refused_to_lie(self):
        assert _RES_T3.search("He refused to lie even to save himself.")

    def test_t3_refused_to_deceive(self):
        assert _RES_T3.search("She refused to deceive the court.")

    def test_t3_would_not_betray(self):
        assert _RES_T3.search("She would not betray the people who trusted her.")

    def test_t3_never_betrayed(self):
        assert _RES_T3.search("He never betrayed his friends.")

    def test_t3_never_abandoned(self):
        assert _RES_T3.search("She never abandoned those in her care.")

    def test_t3_could_not_abandon_them(self):
        assert _RES_T3.search("She could not abandon them in their hour of need.")

    def test_t3_could_not_forsake(self):
        assert _RES_T3.search("He could not forsake the people who depended on him.")

    # --- Neutral text matches nothing ---
    def test_neutral_matches_no_tier(self):
        text = "The project was completed on schedule."
        assert not _RES_T1.search(text)
        assert not _RES_T2.search(text)
        assert not _RES_T3.search(text)


# ---------------------------------------------------------------------------
# TestStakesTiers
# ---------------------------------------------------------------------------

class TestStakesTiers:
    """
    T1: abstract stakes (something at stake, hanging in balance).
    T2: named external agent applying pressure (threatened/warned/ordered to).
    T3: physical violence, death, imprisonment, or execution named.
    """

    # --- T1 matches ---
    def test_t1_life_at_stake(self):
        assert _STK_T1.search("His life was at stake when he made that choice.")

    def test_t1_career_at_stake(self):
        assert _STK_T1.search("Her career was at stake, yet she refused.")

    def test_t1_freedom_depended(self):
        assert _STK_T1.search("Their freedom depended on it.")

    def test_t1_future_hung_in_balance(self):
        assert _STK_T1.search("His future hung in the balance.")

    def test_t1_reputation_at_stake(self):
        assert _STK_T1.search("Her reputation was at stake.")

    # --- T1 phrases do NOT match T2 or T3 ---
    def test_life_at_stake_not_t2(self):
        assert not _STK_T2.search("His life was at stake.")

    def test_career_at_stake_not_t3(self):
        assert not _STK_T3.search("Her career was at stake.")

    # --- T2 matches ---
    def test_t2_threatened_him_to(self):
        assert _STK_T2.search("They threatened him to step aside.")

    def test_t2_warned_them_to(self):
        assert _STK_T2.search("The authorities warned them to stop immediately.")

    def test_t2_ordered_her_to(self):
        assert _STK_T2.search("The commander ordered her to recant.")

    # --- T2 phrases do NOT match T3 ---
    def test_threatened_to_not_t3(self):
        assert not _STK_T3.search("They threatened him to step aside.")

    # --- T3 matches ---
    def test_t3_risked_death(self):
        assert _STK_T3.search("He risked death to carry the message.")

    def test_t3_risked_imprisonment(self):
        assert _STK_T3.search("She risked imprisonment to speak the truth.")

    def test_t3_faced_execution(self):
        assert _STK_T3.search("He faced execution rather than renounce his beliefs.")

    def test_t3_faced_exile(self):
        assert _STK_T3.search("She faced exile for her refusal to comply.")

    def test_t3_threatened_death(self):
        assert _STK_T3.search("They threatened death if he refused to comply.")

    def test_t3_at_gunpoint(self):
        assert _STK_T3.search("He still refused at gunpoint.")

    def test_t3_at_threat_of_death(self):
        assert _STK_T3.search("Even at threat of death she would not recant.")

    def test_t3_at_knifepoint(self):
        assert _STK_T3.search("She refused to confess even at knifepoint.")

    def test_t3_faced_persecution(self):
        assert _STK_T3.search("They faced persecution for their beliefs.")

    # --- Neutral text matches nothing ---
    def test_neutral_matches_no_tier(self):
        text = "The quarterly report showed steady growth."
        assert not _STK_T1.search(text)
        assert not _STK_T2.search(text)
        assert not _STK_T3.search(text)


# ---------------------------------------------------------------------------
# TestClassScore
# ---------------------------------------------------------------------------

class TestClassScore:
    """_class_score() must return the HIGHEST matching tier."""

    def test_returns_t3_when_t3_matches(self):
        # Text matches T3 adversity
        score = _class_score(
            "Risking everything, she spoke out.",
            _ADV_T1, _ADV_T2, _ADV_T3,
        )
        assert score == _T3

    def test_returns_t2_when_only_t2_matches(self):
        score = _class_score(
            "At great risk he continued.",
            _ADV_T1, _ADV_T2, _ADV_T3,
        )
        assert score == _T2

    def test_returns_t1_when_only_t1_matches(self):
        score = _class_score(
            "Despite the difficulty he pressed on.",
            _ADV_T1, _ADV_T2, _ADV_T3,
        )
        assert score == _T1

    def test_returns_zero_when_nothing_matches(self):
        score = _class_score(
            "The meeting was on Tuesday.",
            _ADV_T1, _ADV_T2, _ADV_T3,
        )
        assert score == 0.0

    def test_t3_wins_when_t1_also_matches(self):
        # "despite" (T1) and "risking everything" (T3) both in text
        text = "Despite everything, risking her life, she spoke."
        score = _class_score(text, _ADV_T1, _ADV_T2, _ADV_T3)
        assert score == _T3

    def test_t2_wins_when_t1_also_matches(self):
        # "despite" (T1) and "under pressure" (T2) both in text — T2 wins
        text = "Despite everything, he stood firm under pressure."
        score = _class_score(text, _ADV_T1, _ADV_T2, _ADV_T3)
        assert score == _T2


# ---------------------------------------------------------------------------
# TestStructuralScore
# ---------------------------------------------------------------------------

class TestStructuralScore:
    """
    structural_score = mean of 4 class scores.

    Key calibration points:
      no signals              → 0.0
      1 class T1 only         → 0.075  (0.30 / 4)
      1 class T3 only         → 0.25   (1.00 / 4)
      all 4 classes at T1     → 0.30
      all 4 classes at T2     → 0.65
      all 4 classes at T3     → 1.00
    """

    def test_no_signals_returns_zero(self):
        text = "The meeting was held on Tuesday and the committee adjourned."
        assert structural_score(text) == 0.0

    def test_one_t1_class_returns_point_075(self):
        # "Despite" fires ADV T1 only. No agency/resistance/stakes.
        text = "Despite the weather, the session concluded normally."
        assert structural_score(text) == pytest.approx(0.075, abs=1e-4)

    def test_one_t3_class_returns_point_25(self):
        # "at gunpoint" fires STK T3 only. No adversity/agency/resistance.
        text = "The witness testified at gunpoint."
        assert structural_score(text) == pytest.approx(0.25, abs=1e-4)

    def test_all_four_t1_classes_returns_point_30(self):
        # Each class fires at T1 level.
        # ADV-T1: "despite" | AGC-T1: "I continued" | RES-T1: "did not waver"
        # STK-T1: "her reputation was at stake"
        text = (
            "Despite the adversity, I continued. "
            "She did not waver. "
            "Her reputation was at stake."
        )
        assert structural_score(text) == pytest.approx(_T1, abs=1e-4)

    def test_all_four_t3_classes_returns_one(self):
        # Every class fires at T3.
        text = (
            "Risking everything, I would not betray them. "
            "She refused to abandon those she loved. "
            "They faced execution at gunpoint."
        )
        assert structural_score(text) == 1.0

    def test_t3_adversity_t2_remainder_correct(self):
        # ADV-T3=1.0, AGC-T2=0.65 (refused), RES-T2=0.65 (refused to yield),
        # STK-T1=0.30 (life at stake)
        text = (
            "Risking his life, he refused to yield. "
            "He refused and I stood firm. "
            "His life was at stake."
        )
        score = structural_score(text)
        assert 0.60 < score < 0.80

    def test_score_strictly_increases_with_intensity(self):
        """
        The key improvement: more intense language produces a higher score.
        despite < at great cost < risking everything
        """
        mild     = structural_score("Despite the difficulty.")
        moderate = structural_score("He acted at great cost to himself.")
        extreme  = structural_score("Risking everything, she spoke out.")
        assert mild < moderate < extreme, (
            f"Scores must increase with adversity intensity: "
            f"mild={mild}, moderate={moderate}, extreme={extreme}"
        )

    def test_return_type_is_float(self):
        assert isinstance(structural_score("text"), float)

    def test_score_in_valid_range(self):
        texts = [
            "The report was filed.",
            "Despite all odds she continued.",
            "I refused to yield despite the danger.",
            "Risking everything I would not betray them at gunpoint.",
        ]
        for text in texts:
            s = structural_score(text)
            assert 0.0 <= s <= 1.0, f"structural_score out of range for: {text!r}"

    def test_empty_text_returns_zero(self):
        assert structural_score("") == 0.0

    def test_fail_open_on_none(self):
        assert structural_score(None) == 0.0  # type: ignore

    def test_deterministic(self):
        text = "Despite the danger I refused to yield even at gunpoint."
        assert structural_score(text) == structural_score(text)


# ---------------------------------------------------------------------------
# TestTierDiscrimination
# ---------------------------------------------------------------------------

class TestTierDiscrimination:
    """
    The core invariant of the tiered system: language of greater moral and
    physical intensity must produce a strictly higher structural score.

    These tests would FAIL on the old binary system (where 'despite' and
    'risking everything' produced the same adversity score).
    """

    def test_risking_everything_beats_despite(self):
        s_mild    = structural_score("Despite the difficulty, he told the truth.")
        s_extreme = structural_score("Risking everything, he told the truth.")
        assert s_extreme > s_mild, (
            "Existential adversity must score higher than concessive adversity"
        )

    def test_at_gunpoint_beats_life_at_stake(self):
        s_abstract = structural_score("His life was at stake.")
        s_concrete = structural_score("He was forced at gunpoint.")
        assert s_concrete > s_abstract, (
            "Concrete lethal threat must score higher than abstract stake"
        )

    def test_would_not_betray_beats_continued(self):
        s_passive = structural_score("I continued despite the exhaustion.")
        s_active  = structural_score("I would not betray my comrades.")
        assert s_active > s_passive, (
            "Explicit moral refusal must score higher than passive continuation"
        )

    def test_refused_to_betray_beats_refused_to_yield(self):
        # Both are resistance patterns but different tiers.
        s_t2 = structural_score("He refused to yield.")
        s_t3 = structural_score("He refused to betray his men.")
        assert s_t3 > s_t2, (
            "Moral betrayal refusal (T3) must score higher than positional refusal (T2)"
        )

    def test_full_extreme_passage_near_one(self):
        # First-person required for agency class to fire.
        text = (
            "Risking my life, I would not betray them. "
            "I refused to abandon the wounded. "
            "They faced execution at gunpoint."
        )
        assert structural_score(text) >= 0.90

    def test_pure_concessive_passage_near_zero(self):
        text = "Despite the inconvenience, she continued the meeting."
        assert structural_score(text) < 0.20, (
            "Mild concessive with no other signals must produce a low score"
        )

    def test_adding_t3_signal_raises_score(self):
        """Adding an extreme signal to a mild passage must strictly increase the score."""
        mild    = "Despite the difficulty, he pressed on."
        extreme = mild + " They threatened him with execution."
        assert structural_score(extreme) > structural_score(mild)

    def test_tier_ordering_across_all_classes(self):
        """For each class, T3 text must outscore T1 text."""
        pairs = [
            # (T1 text, T3 text, class_name)
            ("Despite the problem.", "Risking everything he acted.", "adversity"),
            ("I continued working.", "I would not betray them.", "agency"),
            ("He did not falter.", "He refused to betray his men.", "resistance"),
            ("Her career was at stake.", "She faced execution.", "stakes"),
        ]
        for t1_text, t3_text, label in pairs:
            s1 = structural_score(t1_text)
            s3 = structural_score(t3_text)
            assert s3 > s1, (
                f"{label}: T3 text must outscore T1 text "
                f"(T1={s1:.4f}, T3={s3:.4f})"
            )


# ---------------------------------------------------------------------------
# TestZeroshotScores — mocked pipeline
# ---------------------------------------------------------------------------

class TestZeroshotScores:
    """Test the zeroshot_scores() interface using a mock pipeline."""

    def _make_mock_pipe(self, label_score_pairs: list) -> MagicMock:
        labels = [VALUE_HYPOTHESES[v] for v, _ in label_score_pairs]
        scores = [s for _, s in label_score_pairs]
        mock = MagicMock(return_value={"labels": labels, "scores": scores})
        return mock

    def test_returns_empty_when_no_pipeline(self):
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            assert zeroshot_scores("text", ["courage"]) == []

    def test_returns_empty_for_empty_text(self):
        mock_pipe = MagicMock()
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            assert zeroshot_scores("", ["courage"]) == []

    def test_returns_empty_for_empty_values(self):
        mock_pipe = MagicMock()
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            assert zeroshot_scores("brave text", []) == []

    def test_returns_empty_for_unknown_value(self):
        mock_pipe = MagicMock()
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            assert zeroshot_scores("text", ["not_a_real_value"]) == []

    def test_above_threshold_value_returned(self):
        mock_pipe = self._make_mock_pipe([("courage", 0.85)])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores("brave text", ["courage"], threshold=0.50)
        assert len(result) == 1
        assert result[0] == ("courage", 0.85)

    def test_below_threshold_value_excluded(self):
        mock_pipe = self._make_mock_pipe([("courage", 0.40)])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            assert zeroshot_scores("text", ["courage"], threshold=0.50) == []

    def test_results_sorted_by_score_desc(self):
        mock_pipe = self._make_mock_pipe([
            ("courage", 0.60),
            ("integrity", 0.90),
            ("patience", 0.75),
        ])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores(
                "text", ["courage", "integrity", "patience"], threshold=0.30
            )
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_returns_value_name_not_hypothesis_string(self):
        mock_pipe = self._make_mock_pipe([("integrity", 0.95)])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores("text", ["integrity"], threshold=0.50)
        assert result[0][0] == "integrity"
        assert result[0][0] in VALUE_HYPOTHESES

    def test_scores_rounded_to_4_decimal_places(self):
        mock_pipe = self._make_mock_pipe([("courage", 0.12345678)])
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            result = zeroshot_scores("text", ["courage"], threshold=0.10)
        score_str = str(result[0][1])
        if "." in score_str:
            assert len(score_str.split(".")[1]) <= 4

    def test_inference_error_returns_empty(self):
        mock_pipe = MagicMock(side_effect=RuntimeError("OOM"))
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            assert zeroshot_scores("text", ["courage"]) == []


# ---------------------------------------------------------------------------
# TestLayer3Signals
# ---------------------------------------------------------------------------

class TestLayer3Signals:
    def test_returns_triple(self):
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            result = layer3_signals("text", 0.90, "action", [])
        struct_score, new_sigs, agreement = result
        assert isinstance(struct_score, float)
        assert isinstance(new_sigs, list)
        assert isinstance(agreement, dict)

    def test_structural_score_propagates(self):
        strong = (
            "Risking his life, I would not betray them. "
            "She faced execution at gunpoint."
        )
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            score, _, _ = layer3_signals(strong, 0.90, "action", [])
        assert score > 0.0

    def test_extreme_passage_high_structural_score(self):
        """Extreme passage should produce a structural score well above 0.5."""
        extreme = (
            "Risking everything, I would not betray them. "
            "She refused to abandon the wounded. "
            "They faced execution at gunpoint."
        )
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            score, _, _ = layer3_signals(extreme, 0.90, "action", [])
        assert score >= 0.75

    def test_mild_passage_low_structural_score(self):
        """Mild concessive passage should produce a low structural score."""
        mild = "Despite the inconvenience, she attended the meeting."
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            score, _, _ = layer3_signals(mild, 0.90, "action", [])
        assert score < 0.20

    def test_zeroshot_disabled_returns_empty_agreement(self):
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
            _, new_sigs, agreement = layer3_signals(
                "text", 0.90, "action", ["courage"],
                zeroshot_enabled=False,
            )
        assert new_sigs == []
        assert agreement == {}

    def test_agreement_placed_in_correct_dict(self):
        hyp = VALUE_HYPOTHESES["courage"]
        mock_pipe = MagicMock(return_value={"labels": [hyp], "scores": [0.80]})
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, agreement = layer3_signals(
                "brave text", 0.90, "action", ["courage"],
                zeroshot_threshold=0.50,
            )
        assert "courage" in agreement
        assert "courage" not in {s["value_name"] for s in new_sigs}

    def test_standalone_detection_above_threshold(self):
        hyp = VALUE_HYPOTHESES["integrity"]
        mock_pipe = MagicMock(return_value={"labels": [hyp], "scores": [0.85]})
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, _ = layer3_signals(
                "honest text", 0.90, "action", [],
                zeroshot_threshold=0.35,
                zeroshot_standalone_threshold=0.70,
            )
        assert "integrity" in {s["value_name"] for s in new_sigs}

    def test_standalone_below_threshold_excluded(self):
        hyp = VALUE_HYPOTHESES["integrity"]
        mock_pipe = MagicMock(return_value={"labels": [hyp], "scores": [0.60]})
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, _ = layer3_signals(
                "honest text", 0.90, "action", [],
                zeroshot_threshold=0.35,
                zeroshot_standalone_threshold=0.70,
            )
        assert new_sigs == []

    def test_new_signal_has_required_fields(self):
        hyp = VALUE_HYPOTHESES["loyalty"]
        mock_pipe = MagicMock(return_value={"labels": [hyp], "scores": [0.90]})
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, _ = layer3_signals(
                "loyal text", 0.90, "action", [],
                zeroshot_threshold=0.35,
                zeroshot_standalone_threshold=0.70,
            )
        assert new_sigs
        for field in ("value_name", "text_excerpt", "significance",
                      "disambiguation_confidence", "source"):
            assert field in new_sigs[0], f"New signal missing field: {field}"

    def test_new_signal_source_is_zeroshot(self):
        hyp = VALUE_HYPOTHESES["loyalty"]
        mock_pipe = MagicMock(return_value={"labels": [hyp], "scores": [0.90]})
        with patch("core.structural_layer._get_zeroshot_pipeline", return_value=mock_pipe):
            _, new_sigs, _ = layer3_signals(
                "text", 0.90, "action", [],
                zeroshot_threshold=0.35,
                zeroshot_standalone_threshold=0.70,
            )
        assert new_sigs[0]["source"] == "zeroshot"


# ---------------------------------------------------------------------------
# TestRegexEdgeCases
# ---------------------------------------------------------------------------

class TestRegexEdgeCases:
    def test_all_tier_patterns_case_insensitive(self):
        assert _ADV_T1.search("DESPITE the danger")
        assert _ADV_T2.search("UNDER PRESSURE he held")
        assert _ADV_T3.search("RISKING EVERYTHING she acted")
        assert _AGC_T1.search("I CONTINUED despite all.")
        assert _AGC_T2.search("I REFUSED to yield.")
        assert _AGC_T3.search("I WOULD NOT betray them.")
        assert _RES_T1.search("SHE DID NOT WAVER.")
        assert _RES_T2.search("HE REFUSED TO YIELD.")
        assert _RES_T3.search("SHE REFUSED TO BETRAY.")
        assert _STK_T1.search("HIS LIFE WAS AT STAKE.")
        assert _STK_T2.search("THEY WARNED HIM TO.")
        assert _STK_T3.search("HE FACED EXECUTION.")

    def test_value_hypotheses_covers_all_15_values(self):
        from core.value_extractor import VALUE_VOCAB
        vocab_keys = set(VALUE_VOCAB.keys())
        hyp_keys   = set(VALUE_HYPOTHESES.keys())
        assert hyp_keys == vocab_keys, (
            f"VALUE_HYPOTHESES must cover all 15 values.\n"
            f"Missing: {vocab_keys - hyp_keys}\n"
            f"Extra: {hyp_keys - vocab_keys}"
        )

    def test_structural_score_is_deterministic(self):
        text = "Despite the danger I would not betray them even at gunpoint."
        assert structural_score(text) == structural_score(text)

    def test_tier_scores_sum_correctly(self):
        """
        A passage that fires exactly one class at each tier (ADV=T1, AGC=T2,
        RES=T3, STK=0) should equal (0.30 + 0.65 + 1.00 + 0.0) / 4 = 0.4875.
        """
        # ADV-T1: "despite" | AGC-T2: "I refused" | RES-T3: "refused to betray"
        # STK: none
        text = "Despite the difficulty, I refused, and she refused to betray anyone."
        score = structural_score(text)
        # ADV=T1=0.30, AGC=T2=0.65 (refused), RES=T3=1.00 (refused to betray), STK=0
        expected = (0.30 + 0.65 + 1.00 + 0.0) / 4.0
        assert score == pytest.approx(expected, abs=1e-3)
