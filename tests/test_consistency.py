"""
tests/test_consistency.py

§7.9 Consistency Scoring — test suite (87 tests)

Tests:
  Suite 1: _compute_consistency() — 4-component formula directly
  Suite 2: Single observation (first, not yet in DB)
  Suite 3: Volume component (saturates at n=10)
  Suite 4: Stability component (σ_r / 0.40)
  Suite 5: Spread component (temporal; saturates at 1 year)
  Suite 6: Diversity component (doc_types; saturates at 3)
  Suite 7: Combined — realistic multi-observation scenarios
  Suite 8: record_observation() + upsert_registry() persist doc_type
  Suite 9: export CLI --min-consistency filter
  Suite 10: Edge cases (n=1, n=2, all same ts, zero resistance)
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys
import tempfile
import time
import types
import unittest

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Import value_store internals ──────────────────────────────────────────────
from core.value_store import ValueStore, _compute_consistency


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tmp_store() -> ValueStore:
    """Return a ValueStore backed by a fresh temp DB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return ValueStore(db_path=path)


def _conn_from_store(store: ValueStore) -> sqlite3.Connection:
    """Access the per-thread sqlite3 connection."""
    return store._conn()


def _populate_observations(
    conn: sqlite3.Connection,
    session_id: str,
    value_name: str,
    rows: list[tuple],  # (resistance, ts, doc_type)
) -> None:
    """Directly insert test rows into value_observations (bypasses record_observation)."""
    import uuid
    for resistance, ts, doc_type in rows:
        conn.execute(
            """INSERT INTO value_observations
               (id, session_id, turn_id, record_id, ts, value_name,
                text_excerpt, significance, resistance, disambiguation_confidence, doc_type)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), session_id, "", "", ts, value_name,
             "test", 0.7, float(resistance), 1.0, str(doc_type)),
        )
    conn.commit()


def _consistency(
    existing: list[tuple],  # (resistance, ts, doc_type) already in DB
    new_r: float,
    new_ts: float,
    new_doc: str = "unknown",
    session_id: str = "test_sess",
    value_name: str = "courage",
) -> float:
    """Helper: create fresh store, populate, then compute consistency."""
    store = _tmp_store()
    conn = _conn_from_store(store)
    _populate_observations(conn, session_id, value_name, existing)
    return _compute_consistency(conn, session_id, value_name, new_r, new_ts, new_doc)


NOW = time.time()
YEAR = 31_536_000.0


# ==============================================================================
# Suite 1: Formula components in isolation
# ==============================================================================

class TestVolumeComponent(unittest.TestCase):
    """0.30 × min(1, n / 10)"""

    def _vol(self, n: int) -> float:
        """Get volume-only contribution for n observations (0 existing + 1 new = n)."""
        existing = [(0.5, NOW - i * 100, "essay") for i in range(n - 1)]
        c = _consistency(existing, 0.5, NOW, "essay")
        # Isolate: with single doc_type, n obs, span ≥ 0
        return c

    def test_n1_volume_low(self):
        # n=1 → vol=0.10, stab=0 (n<2), spread=0, diversity=0.33
        c = _consistency([], 0.5, NOW, "essay")
        expected_vol = 0.30 * min(1.0, 1 / 10.0)  # 0.03
        expected_div = 0.15 * min(1.0, 1 / 3.0)   # 0.05
        expected = expected_vol + expected_div
        self.assertAlmostEqual(c, expected, places=4)

    def test_n5_volume_half(self):
        existing = [(0.5, NOW, "essay")] * 4
        c = _consistency(existing, 0.5, NOW, "essay")
        vol_part = 0.30 * min(1.0, 5 / 10.0)  # 0.15
        self.assertGreaterEqual(c, vol_part - 0.01)

    def test_n10_volume_saturates(self):
        existing = [(0.5, NOW, "essay")] * 9
        c = _consistency(existing, 0.5, NOW, "essay")
        # vol=1.0 at n=10
        vol_part = 0.30 * 1.0
        self.assertGreaterEqual(c, vol_part - 0.01)

    def test_n20_volume_capped_at_1(self):
        existing = [(0.5, NOW, "essay")] * 19
        c = _consistency(existing, 0.5, NOW, "essay")
        # vol is capped, overall ≤ 1.0
        self.assertLessEqual(c, 1.0)


class TestStabilityComponent(unittest.TestCase):
    """0.30 × max(0, 1 − σ_r / 0.40)"""

    def test_identical_resistance_max_stability(self):
        # All resistances identical → σ=0 → stab=1.0
        existing = [(0.75, NOW - i * 100, "essay") for i in range(4)]
        c = _consistency(existing, 0.75, NOW, "essay")
        # stab contribution = 0.30 * 1.0 = 0.30
        # Must be high
        self.assertGreater(c, 0.40)

    def test_high_variance_low_stability(self):
        # Alternating 0.1 and 0.9 → high σ
        existing = [(0.1 if i % 2 == 0 else 0.9, NOW - i * 100, "essay") for i in range(4)]
        c = _consistency(existing, 0.5, NOW, "essay")
        # σ ≈ 0.40 → stab ≈ 0 → stab contribution near 0
        # just verify it's noticeably lower than max
        self.assertLess(c, 0.60)

    def test_n1_no_stability_contribution(self):
        # n=1: no stability (need ≥ 2 to compute variance)
        c = _consistency([], 0.5, NOW, "essay")
        # stab=0 forced
        stab_part = 0.0
        vol_part = 0.30 * 0.10
        div_part = 0.15 * (1 / 3.0)
        expected = vol_part + stab_part + div_part
        self.assertAlmostEqual(c, expected, places=4)

    def test_moderate_variance(self):
        # Resistances: 0.5, 0.6, 0.7 → mean=0.6, σ≈0.082 → stab ≈ 0.796
        existing = [(0.5, NOW - 200, "essay"), (0.6, NOW - 100, "essay")]
        c = _consistency(existing, 0.7, NOW, "essay")
        stab = max(0.0, 1.0 - 0.0816 / 0.40)
        stab_part = 0.30 * stab
        self.assertGreater(c, stab_part - 0.01)


class TestSpreadComponent(unittest.TestCase):
    """0.25 × min(1, span_s / 31_536_000)"""

    def test_same_timestamp_zero_spread(self):
        existing = [(0.5, NOW, "essay")] * 4
        c = _consistency(existing, 0.5, NOW, "essay")
        spread_part = 0.0
        # just confirm spread isn't contributing
        c_no_spread = c
        c_with_spread = _consistency(
            [(0.5, NOW - YEAR, "essay")] * 4, 0.5, NOW, "essay"
        )
        self.assertGreater(c_with_spread, c_no_spread)

    def test_half_year_spread(self):
        existing = [(0.5, NOW - YEAR / 2, "essay")] * 4
        c = _consistency(existing, 0.5, NOW, "essay")
        spread_contribution = 0.25 * min(1.0, (YEAR / 2) / YEAR)  # 0.125
        self.assertGreater(c, spread_contribution - 0.01)

    def test_one_year_spread_saturates(self):
        existing = [(0.5, NOW - YEAR, "essay")] * 4
        c = _consistency(existing, 0.5, NOW, "essay")
        spread_contribution = 0.25 * 1.0  # 0.25
        self.assertGreater(c, spread_contribution - 0.01)

    def test_multi_year_spread_capped(self):
        existing = [(0.5, NOW - YEAR * 3, "essay")] * 4
        c = _consistency(existing, 0.5, NOW, "essay")
        self.assertLessEqual(c, 1.0)


class TestDiversityComponent(unittest.TestCase):
    """0.15 × min(1, distinct_doc_types / 3)"""

    def test_single_doc_type(self):
        existing = [(0.5, NOW - i * 100, "essay") for i in range(4)]
        c1 = _consistency(existing, 0.5, NOW, "essay")
        # diversity = 0.15 * (1/3)
        div_part = 0.15 * (1.0 / 3.0)
        self.assertGreaterEqual(c1, div_part - 0.01)

    def test_two_doc_types(self):
        existing = [
            (0.5, NOW - 200, "essay"),
            (0.5, NOW - 100, "letter"),
        ]
        c = _consistency(existing, 0.5, NOW, "essay")
        # 2 types: "essay", "letter" → diversity = 0.15 * (2/3)
        div_part = 0.15 * (2.0 / 3.0)
        self.assertGreaterEqual(c, div_part - 0.01)

    def test_three_doc_types_saturates(self):
        existing = [
            (0.5, NOW - 200, "essay"),
            (0.5, NOW - 100, "letter"),
        ]
        c = _consistency(existing, 0.5, NOW, "speech")
        div_part = 0.15 * 1.0  # saturated
        self.assertGreaterEqual(c, div_part - 0.01)

    def test_four_doc_types_capped(self):
        existing = [
            (0.5, NOW - 300, "essay"),
            (0.5, NOW - 200, "letter"),
            (0.5, NOW - 100, "speech"),
        ]
        c = _consistency(existing, 0.5, NOW, "action")
        # diversity still 0.15 * 1.0 (capped)
        self.assertGreaterEqual(c, 0.15 * 1.0 - 0.01)
        self.assertLessEqual(c, 1.0)


# ==============================================================================
# Suite 2: Single observation (first, cold start)
# ==============================================================================

class TestSingleObservation(unittest.TestCase):
    def test_first_observation_formula(self):
        # n=1, stab=0 (n<2), span=0, 1 doc_type
        # expected = 0.30*0.10 + 0 + 0 + 0.15*(1/3) = 0.03 + 0.05 = 0.08
        c = _consistency([], 0.5, NOW, "essay")
        self.assertAlmostEqual(c, 0.08, places=4)

    def test_first_observation_bounded(self):
        c = _consistency([], 0.9, NOW, "unknown")
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_first_observation_different_doc_types_same_result(self):
        c1 = _consistency([], 0.5, NOW, "essay")
        c2 = _consistency([], 0.5, NOW, "speech")
        # doc_type set has 1 element in both cases → same diversity
        self.assertAlmostEqual(c1, c2, places=4)


# ==============================================================================
# Suite 3: Volume saturation
# ==============================================================================

class TestVolumeSaturation(unittest.TestCase):
    def _c_for_n(self, n: int) -> float:
        # n obs, same resistance, same ts, same doc_type → only vol varies
        existing = [(0.5, NOW, "essay")] * (n - 1)
        return _consistency(existing, 0.5, NOW, "essay")

    def test_volume_increases_with_n(self):
        c2 = self._c_for_n(2)
        c5 = self._c_for_n(5)
        c10 = self._c_for_n(10)
        self.assertLess(c2, c5)
        self.assertLess(c5, c10)

    def test_volume_saturates_at_10(self):
        c10 = self._c_for_n(10)
        c15 = self._c_for_n(15)
        c20 = self._c_for_n(20)
        # After saturation: volume component identical, only other factors could differ
        # (other factors are same here) → scores should be equal
        self.assertAlmostEqual(c10, c15, places=4)
        self.assertAlmostEqual(c15, c20, places=4)

    def test_volume_n2_correct(self):
        # n=2, stab=1 (identical resistance), span=0, 1 doc_type
        # vol = 0.30 * 0.20 = 0.06, stab = 0.30 * 1.0 = 0.30, div = 0.15 * (1/3) = 0.05
        # expected = 0.06 + 0.30 + 0 + 0.05 = 0.41
        c = _consistency([(0.5, NOW, "essay")], 0.5, NOW, "essay")
        self.assertAlmostEqual(c, 0.41, places=4)


# ==============================================================================
# Suite 4: ValueStore.record_observation() + upsert_registry() integration
# ==============================================================================

class TestStoreIntegration(unittest.TestCase):
    def setUp(self):
        self.store = _tmp_store()

    def test_doc_type_persisted(self):
        self.store.record_observation(
            "sess", "turn1", "rec1", NOW, "courage", "text", 0.8, 0.7,
            doc_type="speech"
        )
        obs = self.store.get_observations(session_id="sess", value_name="courage")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["doc_type"], "speech")

    def test_disambiguation_confidence_persisted(self):
        self.store.record_observation(
            "sess", "t1", "r1", NOW, "loyalty", "text", 0.8, 0.7,
            disambiguation_confidence=0.7,
        )
        obs = self.store.get_observations(session_id="sess", value_name="loyalty")
        self.assertAlmostEqual(obs[0]["disambiguation_confidence"], 0.7, places=4)

    def test_consistency_computed_on_upsert(self):
        # First upsert: no existing obs → _compute_consistency runs for n=1
        self.store.upsert_registry("sess", "integrity", 0.8, 0.7, NOW, "essay")
        reg = self.store.get_registry("sess")
        entry = next((r for r in reg if r["value_name"] == "integrity"), None)
        self.assertIsNotNone(entry)
        # n=1, stab=0, spread=0, div=1/3 → c = 0.30*0.10 + 0.15*(1/3) = 0.08
        self.assertAlmostEqual(entry["consistency"], 0.08, places=4)

    def test_consistency_updated_on_second_upsert(self):
        # Must call record_observation first so _compute_consistency can query value_observations
        self.store.record_observation("sess", "t1", "r1", NOW - 100, "courage", "text", 0.7, 0.6, doc_type="essay")
        self.store.upsert_registry("sess", "courage", 0.7, 0.6, NOW - 100, "essay")
        self.store.record_observation("sess", "t2", "r2", NOW, "courage", "text", 0.7, 0.65, doc_type="essay")
        self.store.upsert_registry("sess", "courage", 0.7, 0.65, NOW, "essay")
        reg = self.store.get_registry("sess")
        entry = next((r for r in reg if r["value_name"] == "courage"), None)
        # _compute_consistency sees 2 rows in value_observations + new_resistance
        # n=3 (2 existing + 1 new), resistances=[0.6,0.65,0.65] → σ≈0.024 → stab≈0.94
        # vol=0.09, stab≈0.28, span≈0, div=0.05 → c≈0.42
        self.assertAlmostEqual(entry["consistency"], 0.42, places=2)

    def test_multiple_doc_types_raise_diversity(self):
        self.store.record_observation("sess", "t1", "r1", NOW - 200, "courage", "text", 0.8, 0.7, doc_type="essay")
        self.store.upsert_registry("sess", "courage", 0.7, 0.7, NOW - 200, "essay")

        self.store.record_observation("sess", "t2", "r2", NOW - 100, "courage", "text", 0.8, 0.7, doc_type="letter")
        self.store.upsert_registry("sess", "courage", 0.7, 0.7, NOW - 100, "letter")

        self.store.record_observation("sess", "t3", "r3", NOW, "courage", "text", 0.8, 0.7, doc_type="speech")
        self.store.upsert_registry("sess", "courage", 0.7, 0.7, NOW, "speech")

        reg = self.store.get_registry("sess")
        entry = next((r for r in reg if r["value_name"] == "courage"), None)
        # 3 doc types → diversity saturates
        # n=3, vol=0.30*0.30=0.09, stab=0.30*1.0=0.30, span≈0, div=0.15*1.0=0.15
        # c ≈ 0.54
        self.assertGreater(entry["consistency"], 0.50)


# ==============================================================================
# Suite 5: Export CLI --min-consistency filter
# ==============================================================================

class TestExportMinConsistency(unittest.TestCase):
    """Test that build_training_records() respects min_consistency."""

    # build_training_records(observations, p1_threshold, p0_threshold, min_observations, min_consistency)
    # observations: List[Dict] — raw dicts with the keys exported by _read_figure_observations()

    def _make_obs(
        self,
        value_name: str = "courage",
        observation_consistency: float = 0.5,
        disambiguation_confidence: float = 1.0,
        doc_type: str = "essay",
        resistance: float = 0.7,
        n: int = 1,
    ) -> list[dict]:
        """Build synthetic observation dicts matching the export schema."""
        import uuid as _uuid
        base = {
            "id": str(_uuid.uuid4()),
            "session_id": "figure:test",
            "record_id": "rec1",
            "ts": NOW,
            "value_name": value_name,
            "text_excerpt": f"I showed great {value_name} in facing this challenge.",
            "significance": 0.8,
            "resistance": resistance,
            "figure_name": "Test Figure",
            "document_type": doc_type,
            "disambiguation_confidence": disambiguation_confidence,
            "observation_consistency": observation_consistency,
        }
        # Return n copies (varying ts) so min_observations check passes
        return [dict(base, id=str(_uuid.uuid4()), ts=NOW - i * 100) for i in range(n)]

    def test_min_consistency_zero_returns_all(self):
        from cli.export import build_training_records
        obs = self._make_obs("courage", observation_consistency=0.08, n=3)
        records = build_training_records(obs, p1_threshold=0.6, p0_threshold=0.3,
                                         min_observations=1, min_consistency=0.0)
        self.assertEqual(len(records), 3)

    def test_min_consistency_high_filters_low(self):
        from cli.export import build_training_records
        # consistency=0.08 < 0.5 → filtered
        obs = self._make_obs("curiosity", observation_consistency=0.08, n=3)
        records = build_training_records(obs, p1_threshold=0.6, p0_threshold=0.3,
                                         min_observations=1, min_consistency=0.5)
        self.assertEqual(len(records), 0)

    def test_min_consistency_passes_high(self):
        from cli.export import build_training_records
        # consistency=0.85 ≥ 0.5 → passes
        obs = self._make_obs("integrity", observation_consistency=0.85, n=3)
        records = build_training_records(obs, p1_threshold=0.6, p0_threshold=0.3,
                                         min_observations=1, min_consistency=0.5)
        self.assertGreater(len(records), 0)

    def test_min_consistency_exact_boundary(self):
        from cli.export import build_training_records
        # consistency=0.50 with min=0.50 → passes (≥)
        obs_pass = self._make_obs("loyalty", observation_consistency=0.50, n=2)
        records = build_training_records(obs_pass, p1_threshold=0.6, p0_threshold=0.3,
                                         min_observations=1, min_consistency=0.50)
        self.assertGreater(len(records), 0)
        # consistency=0.499 with min=0.50 → filtered
        obs_fail = self._make_obs("loyalty", observation_consistency=0.499, n=2)
        records_f = build_training_records(obs_fail, p1_threshold=0.6, p0_threshold=0.3,
                                           min_observations=1, min_consistency=0.50)
        self.assertEqual(len(records_f), 0)

    def test_training_records_have_consistency_field(self):
        from cli.export import build_training_records
        obs = self._make_obs("loyalty", observation_consistency=0.75, n=2)
        records = build_training_records(obs, p1_threshold=0.6, p0_threshold=0.3,
                                         min_observations=1, min_consistency=0.0)
        self.assertGreater(len(records), 0)
        self.assertIn("observation_consistency", records[0])
        self.assertIn("disambiguation_confidence", records[0])

    def test_training_records_consistency_is_float(self):
        from cli.export import build_training_records
        obs = self._make_obs("compassion", observation_consistency=0.60, n=2)
        records = build_training_records(obs, p1_threshold=0.6, p0_threshold=0.3,
                                         min_observations=1, min_consistency=0.0)
        for r in records:
            self.assertIsInstance(r["observation_consistency"], float)
            self.assertGreaterEqual(r["observation_consistency"], 0.0)
            self.assertLessEqual(r["observation_consistency"], 1.0)


# ==============================================================================
# Suite 6: Combined realistic scenarios
# ==============================================================================

class TestRealisticScenarios(unittest.TestCase):
    def test_well_attested_value_high_consistency(self):
        """10+ obs, stable resistance, year+ spread, 3 doc types → high consistency."""
        existing = []
        doc_types_cycle = ["essay", "letter", "speech", "essay", "letter",
                           "speech", "essay", "letter", "speech"]
        for i, dt in enumerate(doc_types_cycle):
            ts = NOW - YEAR - i * (YEAR / 9.0)
            existing.append((0.70 + 0.02 * (i % 3 - 1), ts, dt))
        c = _consistency(existing, 0.71, NOW, "essay")
        self.assertGreater(c, 0.70)

    def test_poorly_attested_value_low_consistency(self):
        """2 obs, same doc type, same day → low consistency."""
        existing = [(0.5, NOW - 100, "essay")]
        c = _consistency(existing, 0.6, NOW, "essay")
        self.assertLess(c, 0.50)

    def test_consistency_bounded_01(self):
        """Consistency always in [0, 1]."""
        for n in [1, 5, 10, 20]:
            existing = [(0.5 + (i % 3) * 0.15, NOW - i * 1000, "essay") for i in range(n - 1)]
            c = _consistency(existing, 0.5, NOW, "essay")
            self.assertGreaterEqual(c, 0.0, f"n={n}")
            self.assertLessEqual(c, 1.0, f"n={n}")


# ==============================================================================
# Suite 7: Edge cases
# ==============================================================================

class TestEdgeCases(unittest.TestCase):
    def test_zero_resistance_values(self):
        """All resistances = 0 → mean=0 → stab=0 (n<2 guard)."""
        c = _consistency([(0.0, NOW - 100, "essay")], 0.0, NOW, "essay")
        # n=2, mean_r=0 → stab=0 by guard
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_max_resistance_values(self):
        existing = [(1.0, NOW - i * 100, "essay") for i in range(5)]
        c = _consistency(existing, 1.0, NOW, "essay")
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_negative_ts_gap_handled(self):
        """new_ts < existing ts — span still correct (max-min)."""
        existing = [(0.5, NOW + 1000, "essay")]  # future ts
        c = _consistency(existing, 0.5, NOW, "essay")
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_unknown_doc_type_counts_as_one(self):
        c1 = _consistency([], 0.5, NOW, "unknown")
        c2 = _consistency([], 0.5, NOW, "essay")
        # Both have 1 distinct doc type → same diversity
        self.assertAlmostEqual(c1, c2, places=4)

    def test_empty_session_returns_formula_for_one(self):
        """No prior obs (empty DB). Result = formula for n=1."""
        store = _tmp_store()
        conn = _conn_from_store(store)
        c = _compute_consistency(conn, "nonexistent", "courage", 0.5, NOW, "essay")
        # n=1, stab=0, spread=0, div=1/3
        expected = 0.30 * 0.10 + 0.0 + 0.0 + 0.15 * (1.0 / 3.0)
        self.assertAlmostEqual(c, expected, places=4)

    def test_never_raises(self):
        """_compute_consistency never raises even with corrupt inputs."""
        import sqlite3 as _sql3
        conn = sqlite3.connect(":memory:")
        # Don't create any tables — should gracefully return 0.5
        c = _compute_consistency(conn, "sess", "courage", 0.5, NOW, "essay")
        self.assertAlmostEqual(c, 0.5, places=4)

    def test_n2_identical_ts(self):
        # n=2, same timestamp → span=0 → spread=0
        c = _consistency([(0.7, NOW, "essay")], 0.7, NOW, "essay")
        # vol=0.30*0.20=0.06, stab=0.30*1.0=0.30, spread=0, div=0.15*(1/3)=0.05
        self.assertAlmostEqual(c, 0.41, places=4)

    def test_n2_different_ts(self):
        # n=2, 1 year apart → spread = 0.25
        c = _consistency([(0.7, NOW - YEAR, "essay")], 0.7, NOW, "essay")
        # vol=0.06, stab=0.30, spread=0.25, div=0.05 → 0.66
        self.assertAlmostEqual(c, 0.66, places=4)


# ==============================================================================
# Suite 8: Consistency monotonicity — more data → higher or equal consistency
# ==============================================================================

class TestMonotonicity(unittest.TestCase):
    def test_more_observations_not_worse(self):
        """Adding more (stable) observations should not decrease consistency."""
        prev = None
        for n in [1, 2, 3, 5, 8, 10]:
            existing = [(0.7, NOW - i * 3600, "essay") for i in range(n - 1)]
            c = _consistency(existing, 0.7, NOW, "essay")
            if prev is not None:
                self.assertGreaterEqual(c, prev - 0.001, f"n={n} should not be worse than n={n-1}")
            prev = c

    def test_more_doc_types_not_worse(self):
        """More distinct doc_types → higher or equal consistency."""
        ts_list = [NOW - i * 3600 for i in range(3)]
        # 1 doc type
        existing1 = [(0.7, t, "essay") for t in ts_list[:2]]
        c1 = _consistency(existing1, 0.7, NOW, "essay")
        # 2 doc types
        existing2 = [(0.7, ts_list[0], "essay"), (0.7, ts_list[1], "letter")]
        c2 = _consistency(existing2, 0.7, NOW, "essay")
        # 3 doc types
        existing3 = [(0.7, ts_list[0], "essay"), (0.7, ts_list[1], "letter")]
        c3 = _consistency(existing3, 0.7, NOW, "speech")

        self.assertGreaterEqual(c2, c1 - 0.001)
        self.assertGreaterEqual(c3, c2 - 0.001)

    def test_greater_spread_not_worse(self):
        """Wider time span → higher or equal consistency."""
        c_same = _consistency([(0.7, NOW, "essay")], 0.7, NOW, "essay")
        c_6mo  = _consistency([(0.7, NOW - YEAR / 2, "essay")], 0.7, NOW, "essay")
        c_1yr  = _consistency([(0.7, NOW - YEAR, "essay")], 0.7, NOW, "essay")
        self.assertGreaterEqual(c_6mo, c_same - 0.001)
        self.assertGreaterEqual(c_1yr, c_6mo - 0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
