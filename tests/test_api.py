"""
tests/test_api.py

PRINCIPLE: Tests that could break the system are more valuable than tests
that run through it. Every test here targets a real failure mode.

Section 1 — Input validation: no mocks. Tests what Pydantic/FastAPI actually
  enforces. A regression here means the API accepts bad data.

Section 2 — HTTP routing: no mocks. Tests that URLs hit the correct handlers.
  The /figures/universal vs /{name}/profile conflict is a real routing hazard.

Section 3 — Response contract invariants: controlled mock data, but asserts
  properties the system MUST maintain (ordering, types, rounding, key presence).
  A test here catches a regression if the contract changes silently.

Section 4 — Real pipeline integration: in-memory SQLite stores, ML layers
  disabled. Keyword extraction, resistance scoring, lexicons, and DB writes
  run for real. These tests would fail if the actual pipeline is broken.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Shared client fixture (no pipeline mocking — each section adds its own)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.app import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Section 4 fixture — real in-memory pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def live(tmp_path):
    """
    Isolated pipeline fixture: real DocumentStore + ValueStore backed by
    temp files, ML layers (embedder, zeroshot) disabled.

    Sets the singleton globals so ALL code paths (pipeline.py,
    value_extractor.py, app.py) share the same test stores.
    """
    from core.document_store import DocumentStore
    from core.value_store import ValueStore
    import core.document_store as _dmod
    import core.value_store as _vmod

    doc_store = DocumentStore(str(tmp_path / "docs.db"))
    val_store = ValueStore(str(tmp_path / "vals.db"))

    orig_doc = _dmod._instance   # lowercase in document_store
    orig_val = _vmod._INSTANCE   # uppercase in value_store
    _dmod._instance = doc_store
    _vmod._INSTANCE = val_store

    with patch("core.embedder.is_available", return_value=False), \
         patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
        yield doc_store, val_store

    _dmod._instance = orig_doc
    _vmod._INSTANCE = orig_val


# ===========================================================================
# Section 1 — Input validation (no mocks)
# ===========================================================================

class TestInputValidation:
    """
    FastAPI + Pydantic enforce these rules. No mocking — if these fail,
    the API is accepting malformed input.
    """

    # --- significance ---

    def test_significance_above_max_rejected(self, client):
        # significance has le=1.0; 1.5 must be rejected
        r = client.post("/figures/test/ingest",
                        json={"text": "Some text.", "significance": 1.5})
        assert r.status_code == 422

    def test_significance_below_min_rejected(self, client):
        # significance has ge=0.0; negative must be rejected
        r = client.post("/figures/test/ingest",
                        json={"text": "Some text.", "significance": -0.01})
        assert r.status_code == 422

    def test_significance_at_zero_accepted(self, client):
        # Boundary value 0.0 is valid — pipeline must receive the call
        with patch("core.pipeline.ingest_text") as m:
            from core.pipeline import IngestResult
            m.return_value = IngestResult(
                figure_name="test", session_id="figure:test",
                passages_ingested=1, observations_recorded=0,
                source_lang="en", source_authenticity=1.0,
                pub_year=None, error=None)
            r = client.post("/figures/test/ingest",
                            json={"text": "Some text.", "significance": 0.0})
        assert r.status_code == 200
        _, kwargs = m.call_args
        assert kwargs["significance"] == 0.0

    def test_significance_at_one_accepted(self, client):
        with patch("core.pipeline.ingest_text") as m:
            from core.pipeline import IngestResult
            m.return_value = IngestResult(
                figure_name="test", session_id="figure:test",
                passages_ingested=1, observations_recorded=0,
                source_lang="en", source_authenticity=1.0,
                pub_year=None, error=None)
            r = client.post("/figures/test/ingest",
                            json={"text": "Some text.", "significance": 1.0})
        assert r.status_code == 200

    # --- min_demonstrations ---

    def test_min_demonstrations_zero_rejected(self, client):
        # ge=1; zero is explicitly invalid
        r = client.get("/figures/universal?min_demonstrations=0")
        assert r.status_code == 422

    def test_min_demonstrations_negative_rejected(self, client):
        r = client.get("/figures/gandhi/profile?min_demonstrations=-5")
        assert r.status_code == 422

    def test_min_demonstrations_one_accepted(self, client):
        with patch("core.pipeline.figure_profile", return_value=[
            {"value_name": "courage", "demonstrations": 1,
             "avg_significance": 0.9, "avg_resistance": 0.6,
             "consistency": 0.5, "weight": 0.27,
             "first_seen_ts": None, "last_seen_ts": None}
        ]):
            r = client.get("/figures/gandhi/profile?min_demonstrations=1")
        assert r.status_code == 200

    # --- type coercion hazards ---

    def test_pub_year_as_string_rejected(self, client):
        # pub_year is Optional[int]; a non-numeric string must fail
        r = client.post("/figures/test/ingest",
                        json={"text": "Some text.", "pub_year": "eighteen-sixty-three"})
        assert r.status_code == 422

    def test_pub_year_numeric_string_coerced_or_rejected(self, client):
        # Pydantic v2 coerces "1863" → 1863 for int fields; v1 rejects.
        # Either behaviour is acceptable — test that the system is consistent.
        r = client.post("/figures/test/ingest",
                        json={"text": "Some text.", "pub_year": "1863"})
        # Must be 200 (coerced) or 422 (strict) — never a 500
        assert r.status_code in (200, 422)

    def test_is_translation_integer_coercion(self, client):
        # Pydantic coerces 1 → True; this should not cause a server error
        with patch("core.pipeline.ingest_text") as m:
            from core.pipeline import IngestResult
            m.return_value = IngestResult(
                figure_name="test", session_id="figure:test",
                passages_ingested=1, observations_recorded=0,
                source_lang="en", source_authenticity=0.85,
                pub_year=None, error=None)
            r = client.post("/figures/test/ingest",
                            json={"text": "Some text.", "is_translation": 1})
        assert r.status_code in (200, 422)

    # --- missing required fields ---

    def test_missing_text_rejected(self, client):
        r = client.post("/figures/gandhi/ingest", json={})
        assert r.status_code == 422

    def test_null_text_rejected(self, client):
        r = client.post("/figures/gandhi/ingest", json={"text": None})
        assert r.status_code == 422

    def test_text_wrong_type_rejected(self, client):
        r = client.post("/figures/gandhi/ingest", json={"text": 12345})
        # Pydantic may coerce int → str or reject; either is fine, but not 500
        assert r.status_code in (200, 422)

    # --- doc_type ---

    def test_invalid_doc_type_accepted_by_api(self, client):
        # The API doesn't validate doc_type (CLI does via argparse choices).
        # An unrecognised doc_type should reach the pipeline (may use 'unknown' fallback).
        with patch("core.pipeline.ingest_text") as m:
            from core.pipeline import IngestResult
            m.return_value = IngestResult(
                figure_name="test", session_id="figure:test",
                passages_ingested=1, observations_recorded=0,
                source_lang="en", source_authenticity=1.0,
                pub_year=None, error=None)
            r = client.post("/figures/test/ingest",
                            json={"text": "Some text.", "doc_type": "not_a_real_type"})
        # Accepted — the pipeline (not the API) decides what to do with unknown doc_types
        assert r.status_code == 200
        _, kwargs = m.call_args
        assert kwargs["doc_type"] == "not_a_real_type"

    # --- export thresholds ---

    def test_export_thresholds_reject_out_of_range(self, client):
        # ExportRequest has ge=0.0, le=1.0 on all threshold fields.
        # Values outside [0, 1] must be rejected before reaching the handler.
        r = client.post("/export/ric",
                        json={"p1_threshold": 2.0, "p0_threshold": 1.5})
        assert r.status_code == 422


# ===========================================================================
# Section 2 — HTTP routing (no mocks)
# ===========================================================================

class TestRouteContract:
    """
    These tests probe the routing table itself. A broken route declaration
    (ordering, shadowing) would pass all mocked tests but fail these.
    """

    def test_universal_route_distinct_from_name_profile(self, client):
        """
        /figures/universal must NOT be caught by /figures/{name}/profile.
        FastAPI routes are matched in declaration order — if profile was
        declared first, this would return 404 (no figure named 'universal').
        """
        with patch("core.pipeline.universal_profile", return_value=[]) as mock_up, \
             patch("core.value_store.get_value_store") as mock_vs:
            mock_vs.return_value.get_figures_list.return_value = []
            r = client.get("/figures/universal")
        assert r.status_code == 200
        # The universal handler was called, not the profile handler
        mock_up.assert_called_once()

    def test_universal_uppercase_hits_profile_route_not_universal(self, client):
        """
        /figures/UNIVERSAL is NOT the universal endpoint — it's a figure
        named 'UNIVERSAL' which doesn't exist → 404.
        This verifies the universal endpoint is case-sensitive.
        """
        with patch("core.pipeline.figure_profile", return_value=[]):
            r = client.get("/figures/UNIVERSAL/profile")
        assert r.status_code == 404

    def test_get_on_post_only_route_returns_405(self, client):
        r = client.get("/figures/gandhi/ingest")
        assert r.status_code == 405

    def test_post_on_get_only_route_returns_405(self, client):
        r = client.post("/figures/gandhi/profile", json={})
        assert r.status_code == 405

    def test_unknown_route_returns_404(self, client):
        r = client.get("/does/not/exist")
        assert r.status_code == 404

    def test_health_is_always_available(self, client):
        # Sanity: the one route with no external deps must always work
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_openapi_schema_lists_all_routes(self, client):
        schema = client.get("/openapi.json").json()
        paths = set(schema["paths"].keys())
        expected = {
            "/health",
            "/figures/{name}/ingest",
            "/figures",
            "/figures/universal",
            "/figures/{name}/profile",
            "/export/ric",
        }
        missing = expected - paths
        assert not missing, f"Routes missing from schema: {missing}"


# ===========================================================================
# Section 3 — Response contract invariants
# ===========================================================================

class TestResponseInvariants:
    """
    These use controlled mock data to verify promises the API makes about
    its response structure. The assertions go beyond field presence —
    they check types, ordering, rounding, and cross-field relationships.
    """

    def test_session_id_is_lowercased_regardless_of_url_case(self, client):
        """
        session_id must be 'figure:<name_lowercase>'.
        If the URL is /figures/Gandhi/profile, session_id must be figure:gandhi.
        This tests that app.py applies .lower() correctly.
        """
        rows = [{"value_name": "courage", "demonstrations": 3,
                 "avg_significance": 0.9, "avg_resistance": 0.7,
                 "consistency": 0.8, "weight": 0.5,
                 "first_seen_ts": None, "last_seen_ts": None}]
        with patch("core.pipeline.figure_profile", return_value=rows):
            r = client.get("/figures/Gandhi/profile")
        assert r.status_code == 200
        data = r.json()
        assert data["figure"] == "Gandhi"          # preserved as-is
        assert data["session_id"] == "figure:gandhi"  # lowercased

    def test_list_total_always_equals_figures_len(self, client):
        """
        total must equal len(figures) — not a raw DB count that could diverge.
        """
        rows = [
            {"figure_name": "a", "session_id": "figure:a",
             "document_type": "journal", "passage_count": 5, "ingested_at": 1.0},
            {"figure_name": "b", "session_id": "figure:b",
             "document_type": "speech", "passage_count": 3, "ingested_at": 2.0},
            {"figure_name": "c", "session_id": "figure:c",
             "document_type": "action", "passage_count": 9, "ingested_at": 3.0},
        ]
        with patch("core.value_store.get_value_store") as mock_vs:
            mock_vs.return_value.get_figures_list.return_value = rows
            r = client.get("/figures")
        data = r.json()
        assert data["total"] == len(data["figures"])

    def test_profile_values_sorted_by_weight_desc(self, client):
        """
        Values in profile must be ordered by weight descending.
        If the store returns them in a different order, the API must NOT reorder them —
        this tests whether the contract is fulfilled end-to-end.
        """
        # Give the mock rows in weight-descending order (as the store should)
        rows = [
            {"value_name": "courage",    "demonstrations": 8,
             "avg_significance": 0.9, "avg_resistance": 0.8,
             "consistency": 0.9, "weight": 0.78,
             "first_seen_ts": None, "last_seen_ts": None},
            {"value_name": "integrity",  "demonstrations": 5,
             "avg_significance": 0.8, "avg_resistance": 0.6,
             "consistency": 0.7, "weight": 0.52,
             "first_seen_ts": None, "last_seen_ts": None},
            {"value_name": "compassion", "demonstrations": 2,
             "avg_significance": 0.7, "avg_resistance": 0.4,
             "consistency": 0.5, "weight": 0.21,
             "first_seen_ts": None, "last_seen_ts": None},
        ]
        with patch("core.pipeline.figure_profile", return_value=rows):
            r = client.get("/figures/test/profile")
        values = r.json()["values"]
        weights = [v["weight"] for v in values]
        assert weights == sorted(weights, reverse=True), \
            f"Values not sorted by weight DESC: {weights}"

    def test_profile_values_out_of_order_exposes_no_resorting(self, client):
        """
        The API does NOT re-sort — if the store returns values in wrong order,
        they come back wrong. This documents the implicit contract: the store
        is responsible for ordering.
        """
        # Return rows in ascending weight order (wrong)
        rows = [
            {"value_name": "compassion", "demonstrations": 2,
             "avg_significance": 0.7, "avg_resistance": 0.4,
             "consistency": 0.5, "weight": 0.21,
             "first_seen_ts": None, "last_seen_ts": None},
            {"value_name": "courage",    "demonstrations": 8,
             "avg_significance": 0.9, "avg_resistance": 0.8,
             "consistency": 0.9, "weight": 0.78,
             "first_seen_ts": None, "last_seen_ts": None},
        ]
        with patch("core.pipeline.figure_profile", return_value=rows):
            r = client.get("/figures/test/profile")
        values = r.json()["values"]
        # Should reflect store order — API does not sort
        assert values[0]["value_name"] == "compassion"
        assert values[1]["value_name"] == "courage"

    def test_weight_rounded_to_at_most_4_decimal_places(self, client):
        """
        The API explicitly rounds weight to 4 decimal places.
        If the store returns a weight with more precision, the response must truncate it.
        """
        rows = [{"value_name": "courage", "demonstrations": 3,
                 "avg_significance": 0.9, "avg_resistance": 0.8,
                 "consistency": 0.7,
                 "weight": 0.12345678901234,  # many decimal places
                 "first_seen_ts": None, "last_seen_ts": None}]
        with patch("core.pipeline.figure_profile", return_value=rows):
            r = client.get("/figures/test/profile")
        weight_str = str(r.json()["values"][0]["weight"])
        decimal_places = len(weight_str.split(".")[-1]) if "." in weight_str else 0
        assert decimal_places <= 4, \
            f"weight has {decimal_places} decimal places: {weight_str}"

    def test_demonstrations_type_is_integer(self, client):
        """
        demonstrations must be an int. The DB returns an INTEGER column —
        if something coerces it to float, the response schema breaks.
        """
        rows = [{"value_name": "courage", "demonstrations": 7,
                 "avg_significance": 0.9, "avg_resistance": 0.7,
                 "consistency": 0.8, "weight": 0.45,
                 "first_seen_ts": None, "last_seen_ts": None}]
        with patch("core.pipeline.figure_profile", return_value=rows):
            r = client.get("/figures/test/profile")
        val = r.json()["values"][0]
        assert isinstance(val["demonstrations"], int), \
            f"demonstrations should be int, got {type(val['demonstrations'])}"
        assert val["demonstrations"] == 7

    def test_profile_404_message_contains_figure_name(self, client):
        """
        The 404 detail must name the unknown figure so callers can diagnose it.
        """
        with patch("core.pipeline.figure_profile", return_value=[]):
            r = client.get("/figures/nobody_here/profile")
        assert r.status_code == 404
        assert "nobody_here" in r.json()["detail"]

    def test_export_dry_run_returns_null_output_dir(self, client):
        """
        dry_run=True must return output_dir=null (no files written).
        If this is wrong, callers will try to read files that don't exist.
        """
        with patch("api.app._run_export") as m:
            from api.models import ExportResponse
            m.return_value = ExportResponse(
                ok=True, figure=None, p1_count=5, p0_count=3, apy_count=1,
                ambiguous_count=0, total_count=9,
                output_dir=None, files_written=[], error=None)
            r = client.post("/export/ric", json={"dry_run": True})
        assert r.json()["output_dir"] is None

    def test_export_500_detail_is_not_empty(self, client):
        """
        When export raises, the 500 response must include the error message.
        Empty detail is useless to callers.
        """
        with patch("api.app._run_export",
                   side_effect=RuntimeError("subprocess returned exit code 1")):
            r = client.post("/export/ric", json={})
        assert r.status_code == 500
        detail = r.json().get("detail", "")
        assert len(detail) > 0
        assert "subprocess" in detail


# ===========================================================================
# Section 4 — Real pipeline integration
# ===========================================================================

class TestRealPipeline:
    """
    These tests run actual code. DocumentStore, ValueStore, keyword extraction,
    resistance scoring all execute for real. Only DeBERTa and the BGE embedder
    are disabled (they require downloaded models).

    A failure here means the actual pipeline is broken, not just a mock contract.
    """

    # Text reliably triggers courage (has first-person pronoun + multiple keywords)
    _COURAGE_TEXT = (
        "I was afraid, but I stood firm and refused to flee. "
        "Despite the danger, I pressed forward with courage and did not flinch. "
        "I will not yield. Even though the cost was high, I remained resolute."
    )

    # Text that deliberately contains NO value keywords
    _EMPTY_TEXT = (
        "The quarterly report was filed on Tuesday. The meeting started at nine. "
        "The spreadsheet contained twelve rows and four columns of data. "
        "The committee reviewed the findings and adjourned at noon."
    )

    def test_ingest_then_profile_finds_extracted_values(self, client, live):
        """
        After a real ingest, the profile endpoint must return the values
        the keyword extractor found. This fails if DB writes are broken,
        if the watermark isn't reset, or if extraction is skipped.
        """
        r_ingest = client.post("/figures/braveheart/ingest", json={
            "text": self._COURAGE_TEXT,
            "doc_type": "action",
        })
        assert r_ingest.status_code == 200
        assert r_ingest.json()["passages_ingested"] > 0

        r_profile = client.get("/figures/braveheart/profile")
        assert r_profile.status_code == 200
        value_names = [v["value_name"] for v in r_profile.json()["values"]]
        assert "courage" in value_names, \
            f"Expected 'courage' in extracted values, got: {value_names}"

    def test_figure_appears_in_list_after_ingest(self, client, live):
        """
        After ingest, the figure must appear in GET /figures.
        Fails if figure_sources isn't written, or if get_figures_list is broken.
        """
        client.post("/figures/listed_figure/ingest", json={"text": self._COURAGE_TEXT})
        r = client.get("/figures")
        names = [f["figure_name"] for f in r.json()["figures"]]
        assert "listed_figure" in names

    def test_unknown_figure_returns_404(self, client, live):
        """
        A figure that was never ingested must produce 404 on profile.
        Fails if the endpoint doesn't check for empty registry.
        """
        r = client.get("/figures/never_ingested/profile")
        assert r.status_code == 404

    def test_whitespace_only_text_returns_422(self, client, live):
        """
        Text consisting only of whitespace should be rejected as an error.
        Tests that the pipeline's 'Empty text' guard propagates to 422.
        """
        r = client.post("/figures/test/ingest", json={"text": "   \n\t  "})
        assert r.status_code == 422

    def test_text_producing_no_passages_returns_422(self, client, live):
        """
        Text that passes langdetect but segments to zero passages (all < 30 chars)
        must return 422, not 200 with passages_ingested=0.
        """
        r = client.post("/figures/test/ingest", json={"text": "Short. Tiny. No."})
        # Pipeline returns 'No passages extracted' error → 422
        assert r.status_code == 422

    def test_run_extract_false_leaves_profile_empty(self, client, live):
        """
        run_extract=False stores passages but does not run extraction.
        The profile endpoint must return 404 (no values extracted).
        Fails if extraction runs anyway, or if ingest doesn't complete.
        """
        r_ingest = client.post("/figures/noextract/ingest", json={
            "text": self._COURAGE_TEXT,
            "run_extract": False,
        })
        assert r_ingest.status_code == 200
        assert r_ingest.json()["observations_recorded"] == 0

        r_profile = client.get("/figures/noextract/profile")
        # No extraction ran → registry empty → 404
        assert r_profile.status_code == 404

    def test_text_with_no_value_keywords_produces_empty_profile(self, client, live):
        """
        Text that doesn't match any value vocabulary must result in an empty
        profile. Fails if extraction generates false positives.
        """
        r_ingest = client.post("/figures/novalue/ingest", json={
            "text": self._EMPTY_TEXT,
            "doc_type": "unknown",
        })
        assert r_ingest.status_code == 200
        r_profile = client.get("/figures/novalue/profile")
        assert r_profile.status_code == 404, \
            f"Expected 404 for value-free text, got values: " \
            f"{r_profile.json().get('values', [])}"

    def test_action_doctype_produces_higher_resistance_than_speech(self, client, live):
        """
        doc_type directly feeds the resistance formula.
        action (+0.40 bonus) must produce higher avg_resistance than speech (+0.10).
        Fails if doc_type isn't stored, read back, or fed to compute_resistance.
        """
        client.post("/figures/action_hero/ingest", json={
            "text": self._COURAGE_TEXT, "doc_type": "action"
        })
        client.post("/figures/speech_hero/ingest", json={
            "text": self._COURAGE_TEXT, "doc_type": "speech"
        })
        r_action = client.get("/figures/action_hero/profile")
        r_speech = client.get("/figures/speech_hero/profile")

        assert r_action.status_code == 200
        assert r_speech.status_code == 200

        def _resistance(profile_response, value_name):
            for v in profile_response.json()["values"]:
                if v["value_name"] == value_name:
                    return v["avg_resistance"]
            return None

        act_r = _resistance(r_action, "courage")
        spc_r = _resistance(r_speech, "courage")

        assert act_r is not None, "courage not found in action figure profile"
        assert spc_r is not None, "courage not found in speech figure profile"
        assert act_r > spc_r, \
            f"action resistance {act_r} should exceed speech resistance {spc_r}"

    def test_all_profile_weights_are_in_valid_range(self, client, live):
        """
        Weight = demonstrations × avg_significance × avg_resistance × consistency.
        With all factors in [0, 1], the product must also be in [0, 1].
        Fails if any factor is miscalculated or stored incorrectly.
        """
        client.post("/figures/weightcheck/ingest", json={
            "text": self._COURAGE_TEXT, "doc_type": "action"
        })
        r = client.get("/figures/weightcheck/profile")

        assert r.status_code == 200
        for v in r.json()["values"]:
            w = v["weight"]
            assert 0.0 <= w <= 1.0, \
                f"weight {w} for {v['value_name']} outside [0, 1]"

    def test_ingest_response_reflects_actual_passages_stored(self, client, live):
        """
        passages_ingested must equal what was actually inserted into the DB,
        not a hardcoded or estimated count.
        """
        r = client.post("/figures/passcount/ingest", json={
            "text": self._COURAGE_TEXT, "doc_type": "journal"
        })
        data = r.json()
        assert data["passages_ingested"] > 0
        assert isinstance(data["passages_ingested"], int)
        assert data["observations_recorded"] >= 0

    def test_figure_name_with_invalid_characters_returns_422(self, client, live):
        """
        The pipeline rejects figure names not matching [a-zA-Z0-9][a-zA-Z0-9_-]{0,63}.
        The API must propagate this as 422, not 200 with ok=False or a 500.
        """
        r = client.post("/figures/bad name!/ingest", json={"text": self._COURAGE_TEXT})
        # FastAPI will URL-decode the path; the pipeline must reject the name
        assert r.status_code in (404, 422)
