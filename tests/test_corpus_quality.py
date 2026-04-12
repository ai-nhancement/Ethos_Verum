"""
tests/test_corpus_quality.py

Tests for the corpus quality gate:
  - core/corpus.py  — get_figure_corpus_quality(), get_all_corpus_quality()
  - core/value_store.py — register_figure_document() deduplication
  - cli/export.py   — gate filtering (blocked / warned / pass-through)

Test design:
  - corpus quality functions are tested against a real SQLite DB (tmp_path)
    populated by directly inserting rows into figure_documents.
  - register_figure_document tests use a real ValueStore instance.
  - export gate tests mock get_figure_corpus_quality to control tier outcomes
    and verify that blocked-figure observations are removed from output.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.corpus import (
    get_figure_corpus_quality,
    get_all_corpus_quality,
    CORPUS_MIN_DOCS_CONFIDENT,
    CORPUS_MIN_TYPES_CONFIDENT,
    CORPUS_MIN_DOCS_PARTIAL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def val_db(tmp_path):
    """
    Minimal ValueStore DB with only the figure_documents (and figure_sources)
    tables initialised. Returns (db_path_str, ValueStore_instance).
    """
    from core.value_store import ValueStore
    vs = ValueStore(str(tmp_path / "vals.db"))
    return str(tmp_path / "vals.db"), vs


def _insert_doc(db_path: str, figure_name: str, doc_title: str,
                doc_type: str, passage_count: int = 5) -> None:
    """Insert one row into figure_documents directly (bypassing ValueStore)."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO figure_documents "
        "(figure_name, doc_title, doc_type, passage_count, ingested_at) "
        "VALUES (?,?,?,?,?)",
        (figure_name, doc_title, doc_type, passage_count, time.time()),
    )
    conn.commit()
    conn.close()


def _insert_figure_source(db_path: str, figure_name: str,
                           doc_type: str = "unknown") -> None:
    """Insert a legacy figure_sources row (no figure_documents entry)."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR IGNORE INTO figure_sources "
        "(figure_name, session_id, document_type, passage_count, ingested_at) "
        "VALUES (?,?,?,?,?)",
        (figure_name, f"figure:{figure_name}:0", doc_type, 0, time.time()),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tier determination — get_figure_corpus_quality()
# ---------------------------------------------------------------------------

class TestTierDetermination:
    """
    CORPUS_MIN_DOCS_CONFIDENT = 3, CORPUS_MIN_TYPES_CONFIDENT = 2,
    CORPUS_MIN_DOCS_PARTIAL   = 2
    """

    def test_zero_docs_returns_preliminary(self, val_db):
        db_path, _ = val_db
        q = get_figure_corpus_quality(db_path, "nobody")
        assert q["confidence_tier"] == "preliminary"

    def test_zero_docs_not_approved(self, val_db):
        # approved_for_export=False even for legacy 0-doc figures — the export gate
        # separately special-cases docs==0 and allows them through; callers that check
        # approved_for_export directly will still see False for these figures.
        db_path, _ = val_db
        q = get_figure_corpus_quality(db_path, "nobody")
        assert q["approved_for_export"] is False

    def test_one_doc_preliminary(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_a", "Speech A", "speech")
        q = get_figure_corpus_quality(db_path, "fig_a")
        assert q["confidence_tier"] == "preliminary"
        assert q["approved_for_export"] is False

    def test_two_docs_same_type_partial(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_b", "Speech A", "speech")
        _insert_doc(db_path, "fig_b", "Speech B", "speech")
        q = get_figure_corpus_quality(db_path, "fig_b")
        assert q["confidence_tier"] == "partial"
        assert q["approved_for_export"] is False

    def test_two_docs_different_types_still_partial(self, val_db):
        # 2 docs with 2 types is partial, not confident (need ≥3 docs)
        db_path, _ = val_db
        _insert_doc(db_path, "fig_c", "Speech A", "speech")
        _insert_doc(db_path, "fig_c", "Journal A", "journal")
        q = get_figure_corpus_quality(db_path, "fig_c")
        assert q["confidence_tier"] == "partial"
        assert q["approved_for_export"] is False

    def test_three_docs_one_type_is_partial(self, val_db):
        # 3 docs but only 1 type — not confident (need ≥2 types)
        db_path, _ = val_db
        for i in range(3):
            _insert_doc(db_path, "fig_d", f"Speech {i}", "speech")
        q = get_figure_corpus_quality(db_path, "fig_d")
        assert q["confidence_tier"] == "partial"
        assert q["approved_for_export"] is False

    def test_three_docs_two_types_confident(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_e", "Speech A",  "speech")
        _insert_doc(db_path, "fig_e", "Speech B",  "speech")
        _insert_doc(db_path, "fig_e", "Journal A", "journal")
        q = get_figure_corpus_quality(db_path, "fig_e")
        assert q["confidence_tier"] == "confident"
        assert q["approved_for_export"] is True

    def test_three_docs_three_types_confident(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_f", "Speech A",  "speech")
        _insert_doc(db_path, "fig_f", "Journal A", "journal")
        _insert_doc(db_path, "fig_f", "Letter A",  "letter")
        q = get_figure_corpus_quality(db_path, "fig_f")
        assert q["confidence_tier"] == "confident"
        assert q["approved_for_export"] is True

    def test_four_docs_two_types_confident(self, val_db):
        db_path, _ = val_db
        for i in range(3):
            _insert_doc(db_path, "fig_g", f"Speech {i}", "speech")
        _insert_doc(db_path, "fig_g", "Journal A", "journal")
        q = get_figure_corpus_quality(db_path, "fig_g")
        assert q["confidence_tier"] == "confident"
        assert q["approved_for_export"] is True

    def test_unknown_figure_returns_preliminary_with_zero_docs(self, val_db):
        db_path, _ = val_db
        q = get_figure_corpus_quality(db_path, "figure_that_does_not_exist")
        assert q["confidence_tier"] == "preliminary"
        assert q["document_count"] == 0
        assert q["approved_for_export"] is False

    def test_never_raises_on_missing_db(self, tmp_path):
        result = get_figure_corpus_quality(str(tmp_path / "ghost.db"), "anybody")
        assert isinstance(result, dict)
        assert result["confidence_tier"] == "preliminary"
        assert result["approved_for_export"] is False


# ---------------------------------------------------------------------------
# Return structure — get_figure_corpus_quality()
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_all_expected_keys_present(self, val_db):
        db_path, _ = val_db
        q = get_figure_corpus_quality(db_path, "fig")
        expected = {
            "figure_name", "document_count", "doc_types",
            "distinct_doc_type_count", "confidence_tier",
            "approved_for_export", "documents", "notes",
        }
        assert expected.issubset(q.keys())

    def test_figure_name_echoed(self, val_db):
        db_path, _ = val_db
        q = get_figure_corpus_quality(db_path, "aristotle")
        assert q["figure_name"] == "aristotle"

    def test_document_count_matches_inserts(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_h", "Doc A", "speech")
        _insert_doc(db_path, "fig_h", "Doc B", "journal")
        q = get_figure_corpus_quality(db_path, "fig_h")
        assert q["document_count"] == 2

    def test_doc_types_list_is_sorted(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_i", "Speech A",  "speech")
        _insert_doc(db_path, "fig_i", "Action A",  "action")
        _insert_doc(db_path, "fig_i", "Journal A", "journal")
        q = get_figure_corpus_quality(db_path, "fig_i")
        assert q["doc_types"] == sorted(q["doc_types"])

    def test_distinct_doc_type_count_is_accurate(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_j", "Speech A", "speech")
        _insert_doc(db_path, "fig_j", "Speech B", "speech")
        _insert_doc(db_path, "fig_j", "Journal A", "journal")
        q = get_figure_corpus_quality(db_path, "fig_j")
        assert q["distinct_doc_type_count"] == 2

    def test_documents_list_has_expected_fields(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_k", "Doc A", "speech", passage_count=10)
        q = get_figure_corpus_quality(db_path, "fig_k")
        assert len(q["documents"]) == 1
        doc = q["documents"][0]
        assert "doc_title" in doc
        assert "doc_type" in doc
        assert "passage_count" in doc
        assert "ingested_at" in doc
        assert doc["doc_title"] == "Doc A"
        assert doc["doc_type"] == "speech"
        assert doc["passage_count"] == 10

    def test_documents_ordered_by_ingested_at(self, val_db):
        db_path, _ = val_db
        # Insert with distinct ingested_at values
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO figure_documents (figure_name,doc_title,doc_type,passage_count,ingested_at)"
            " VALUES (?,?,?,?,?)", ("fig_l", "First", "speech", 5, 1000.0))
        conn.execute(
            "INSERT INTO figure_documents (figure_name,doc_title,doc_type,passage_count,ingested_at)"
            " VALUES (?,?,?,?,?)", ("fig_l", "Second", "journal", 5, 2000.0))
        conn.commit(); conn.close()
        q = get_figure_corpus_quality(db_path, "fig_l")
        titles = [d["doc_title"] for d in q["documents"]]
        assert titles == ["First", "Second"]


# ---------------------------------------------------------------------------
# Notes content — get_figure_corpus_quality()
# ---------------------------------------------------------------------------

class TestNotesContent:
    def test_zero_docs_note_mentions_doc_title(self, val_db):
        db_path, _ = val_db
        q = get_figure_corpus_quality(db_path, "ghost_fig")
        # doc_count == 0 → special note
        assert q["notes"]
        assert "--doc-title" in q["notes"][0]

    def test_one_doc_note_mentions_count_and_type_needed(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_m", "Doc A", "speech")
        q = get_figure_corpus_quality(db_path, "fig_m")
        # Need 2 more docs and 1 more type
        assert q["notes"]
        note = q["notes"][0]
        assert "2 more document(s)" in note
        assert "1 more doc-type(s)" in note

    def test_two_docs_same_type_note_mentions_both_needs(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_n", "Doc A", "speech")
        _insert_doc(db_path, "fig_n", "Doc B", "speech")
        q = get_figure_corpus_quality(db_path, "fig_n")
        note = q["notes"][0]
        assert "1 more document(s)" in note
        assert "1 more doc-type(s)" in note

    def test_two_docs_two_types_note_mentions_only_doc_need(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_o", "Doc A", "speech")
        _insert_doc(db_path, "fig_o", "Doc B", "journal")
        q = get_figure_corpus_quality(db_path, "fig_o")
        note = q["notes"][0]
        assert "1 more document(s)" in note
        assert "doc-type" not in note  # type requirement already met

    def test_three_docs_one_type_note_mentions_type_need(self, val_db):
        db_path, _ = val_db
        for i in range(3):
            _insert_doc(db_path, "fig_p", f"Doc {i}", "speech")
        q = get_figure_corpus_quality(db_path, "fig_p")
        note = q["notes"][0]
        assert "1 more doc-type(s)" in note

    def test_confident_has_no_notes(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "fig_q", "Speech A",  "speech")
        _insert_doc(db_path, "fig_q", "Speech B",  "speech")
        _insert_doc(db_path, "fig_q", "Journal A", "journal")
        q = get_figure_corpus_quality(db_path, "fig_q")
        assert q["notes"] == []


# ---------------------------------------------------------------------------
# get_all_corpus_quality()
# ---------------------------------------------------------------------------

class TestGetAllCorpusQuality:
    def test_empty_db_returns_empty_list(self, val_db):
        db_path, _ = val_db
        result = get_all_corpus_quality(db_path)
        assert result == []

    def test_returns_one_entry_per_figure(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "plato",     "Works A", "journal")
        _insert_doc(db_path, "aristotle", "Works A", "letter")
        result = get_all_corpus_quality(db_path)
        names = {r["figure_name"] for r in result}
        assert names == {"plato", "aristotle"}

    def test_sorted_by_document_count_desc(self, val_db):
        db_path, _ = val_db
        _insert_doc(db_path, "one_doc",   "Doc A", "speech")
        _insert_doc(db_path, "three_docs", "Doc A", "speech")
        _insert_doc(db_path, "three_docs", "Doc B", "journal")
        _insert_doc(db_path, "three_docs", "Doc C", "action")
        result = get_all_corpus_quality(db_path)
        counts = [r["document_count"] for r in result]
        assert counts == sorted(counts, reverse=True)

    def test_includes_legacy_figures_with_zero_docs(self, val_db):
        db_path, _ = val_db
        # Legacy figure: in figure_sources but NOT in figure_documents
        _insert_figure_source(db_path, "legacy_fig", "action")
        result = get_all_corpus_quality(db_path)
        names = {r["figure_name"] for r in result}
        assert "legacy_fig" in names
        legacy = next(r for r in result if r["figure_name"] == "legacy_fig")
        assert legacy["document_count"] == 0
        assert legacy["confidence_tier"] == "preliminary"

    def test_legacy_figure_at_end_after_sorted(self, val_db):
        db_path, _ = val_db
        _insert_figure_source(db_path, "legacy_fig")
        _insert_doc(db_path, "modern_fig", "Doc A", "speech")
        _insert_doc(db_path, "modern_fig", "Doc B", "journal")
        _insert_doc(db_path, "modern_fig", "Doc C", "action")
        result = get_all_corpus_quality(db_path)
        # modern_fig (3 docs) should appear before legacy_fig (0 docs)
        names = [r["figure_name"] for r in result]
        assert names.index("modern_fig") < names.index("legacy_fig")

    def test_never_raises_on_missing_db(self, tmp_path):
        result = get_all_corpus_quality(str(tmp_path / "ghost.db"))
        assert isinstance(result, list)

    def test_figure_in_both_tables_not_returned_twice(self, val_db):
        """A figure with rows in both figure_documents and figure_sources appears once."""
        db_path, vs = val_db
        # Register via ValueStore (puts row in figure_sources)
        vs.register_figure_source("figure:dup_fig:0", "dup_fig", "speech", 5)
        # Also add a figure_documents entry
        _insert_doc(db_path, "dup_fig", "Speech A", "speech")
        _insert_doc(db_path, "dup_fig", "Journal A", "journal")
        _insert_doc(db_path, "dup_fig", "Letter A",  "letter")

        result = get_all_corpus_quality(db_path)
        dup_entries = [r for r in result if r["figure_name"] == "dup_fig"]
        assert len(dup_entries) == 1, (
            f"Expected 1 entry for dup_fig, got {len(dup_entries)}: {dup_entries}"
        )


# ---------------------------------------------------------------------------
# register_figure_document() — deduplication
# ---------------------------------------------------------------------------

class TestRegisterFigureDocument:
    def test_basic_registration_creates_row(self, val_db):
        db_path, vs = val_db
        vs.register_figure_document("socrates", "speech", "Apology", passage_count=12)
        q = get_figure_corpus_quality(db_path, "socrates")
        assert q["document_count"] == 1
        assert q["documents"][0]["doc_title"] == "Apology"
        assert q["documents"][0]["passage_count"] == 12

    def test_duplicate_title_is_skipped(self, val_db):
        db_path, vs = val_db
        vs.register_figure_document("socrates", "speech", "Apology")
        vs.register_figure_document("socrates", "speech", "Apology")  # duplicate
        q = get_figure_corpus_quality(db_path, "socrates")
        assert q["document_count"] == 1  # still 1, not 2

    def test_duplicate_title_logged_as_warning(self, val_db, caplog):
        _, vs = val_db
        import logging
        with caplog.at_level(logging.WARNING, logger="core.value_store"):
            vs.register_figure_document("plato", "journal", "Republic")
            vs.register_figure_document("plato", "journal", "Republic")
        assert any("already ingested" in r.message for r in caplog.records)

    def test_empty_title_allows_multiple_inserts(self, val_db):
        # Untitled documents cannot be deduplicated — both should be inserted
        db_path, vs = val_db
        vs.register_figure_document("aristotle", "action", doc_title="")
        vs.register_figure_document("aristotle", "action", doc_title="")
        q = get_figure_corpus_quality(db_path, "aristotle")
        assert q["document_count"] == 2

    def test_same_title_different_figures_both_allowed(self, val_db):
        db_path, vs = val_db
        vs.register_figure_document("lincoln",    "speech", "Gettysburg Address")
        vs.register_figure_document("washington", "speech", "Gettysburg Address")
        q_lincoln    = get_figure_corpus_quality(db_path, "lincoln")
        q_washington = get_figure_corpus_quality(db_path, "washington")
        assert q_lincoln["document_count"] == 1
        assert q_washington["document_count"] == 1

    def test_different_titles_same_figure_both_inserted(self, val_db):
        db_path, vs = val_db
        vs.register_figure_document("caesar", "action", "Gallic Wars vol 1")
        vs.register_figure_document("caesar", "action", "Gallic Wars vol 2")
        q = get_figure_corpus_quality(db_path, "caesar")
        assert q["document_count"] == 2

    def test_doc_type_recorded_correctly(self, val_db):
        db_path, vs = val_db
        vs.register_figure_document("marcus", "journal", "Meditations Book I")
        q = get_figure_corpus_quality(db_path, "marcus")
        assert q["documents"][0]["doc_type"] == "journal"

    def test_passage_count_recorded_correctly(self, val_db):
        db_path, vs = val_db
        vs.register_figure_document("epictetus", "letter", "Enchiridion", passage_count=47)
        q = get_figure_corpus_quality(db_path, "epictetus")
        assert q["documents"][0]["passage_count"] == 47

    def test_corpus_reaches_confident_tier_after_three_docs(self, val_db):
        db_path, vs = val_db
        vs.register_figure_document("mandela", "speech",  "Dock Statement")
        vs.register_figure_document("mandela", "speech",  "Nobel Lecture")
        vs.register_figure_document("mandela", "journal", "Prison Notes")
        q = get_figure_corpus_quality(db_path, "mandela")
        assert q["confidence_tier"] == "confident"
        assert q["approved_for_export"] is True


# ---------------------------------------------------------------------------
# Export corpus gate — filter behaviour
# ---------------------------------------------------------------------------

class TestExportCorpusGate:
    """
    Tests the gate logic in cli/export.py: blocked figures are removed from
    the observations list; partial figures pass through with a warning;
    confident figures pass through silently.

    Uses mock to control tier without needing full DB setup.
    """

    @pytest.fixture
    def obs_db(self, tmp_path):
        """
        ValueStore with observations for three figures:
          - blocked_fig   (preliminary, docs > 0)
          - warned_fig    (partial)
          - ok_fig        (confident)
        Returns val_db_path.
        """
        from core.value_store import ValueStore
        vs = ValueStore(str(tmp_path / "vals.db"))

        for fig in ("blocked_fig", "warned_fig", "ok_fig"):
            vs.register_figure_source(f"figure:{fig}:0", fig, "speech", 1)
            vs.record_observation(
                session_id=f"figure:{fig}:0",
                turn_id="t0",
                record_id=f"rec_{fig}_1",
                ts=time.time(),
                value_name="courage",
                significance=0.7,
                resistance=0.6,
                doc_type="speech",
                text_excerpt="He fought for justice.",
            )
        return str(tmp_path / "vals.db")

    def _quality(self, tier: str, docs: int, types: int = 2, notes: list = None):
        return {
            "confidence_tier": tier,
            "document_count":  docs,
            "distinct_doc_type_count": types,
            "notes": notes or [],
        }

    def test_blocked_figure_observations_removed(self, obs_db):
        from cli.export import export as run_export, build_training_records as _orig_build

        def fake_quality(db_path, fig):
            if fig == "blocked_fig":
                return self._quality("preliminary", docs=1, types=1)
            return self._quality("confident", docs=3)

        captured_obs = []

        def capturing_build(obs, *a, **kw):
            captured_obs.extend(obs)
            return _orig_build(obs, *a, **kw)

        with patch("core.corpus.get_figure_corpus_quality", side_effect=fake_quality), \
             patch("cli.export.build_training_records", side_effect=capturing_build):
            run_export(
                db_path=obs_db,
                output_dir=None,
                dry_run=True,
                require_corpus_quality=True,
                force=False,
            )

        figure_names = {o.get("figure_name") for o in captured_obs}
        assert "blocked_fig" not in figure_names, (
            f"blocked_fig observations should have been removed by gate, got: {figure_names}"
        )
        assert {"warned_fig", "ok_fig"} <= figure_names, (
            f"Non-blocked figures should pass through, got: {figure_names}"
        )

    def test_force_keeps_blocked_figures_in_output(self, obs_db):
        """
        force=True bypasses the gate FILTER — all figures' observations reach
        build_training_records regardless of their quality tier.
        """
        from cli.export import export as run_export, build_training_records as _orig_build

        def all_preliminary(db_path, fig):
            return self._quality("preliminary", docs=1)

        captured_obs = []

        def capturing_build(obs, *a, **kw):
            captured_obs.extend(obs)
            return _orig_build(obs, *a, **kw)

        with patch("core.corpus.get_figure_corpus_quality", side_effect=all_preliminary), \
             patch("cli.export.build_training_records", side_effect=capturing_build):
            run_export(
                db_path=obs_db,
                output_dir=None,
                dry_run=True,
                require_corpus_quality=True,
                force=True,
            )

        figure_names = {o.get("figure_name") for o in captured_obs}
        assert {"blocked_fig", "warned_fig", "ok_fig"} <= figure_names, (
            f"force=True should pass all figures through gate, got: {figure_names}"
        )

    def test_no_corpus_gate_keeps_blocked_figures_in_output(self, obs_db):
        """require_corpus_quality=False: gate filter skipped entirely."""
        from cli.export import export as run_export, build_training_records as _orig_build

        captured_obs = []

        def capturing_build(obs, *a, **kw):
            captured_obs.extend(obs)
            return _orig_build(obs, *a, **kw)

        with patch("cli.export.build_training_records", side_effect=capturing_build):
            run_export(
                db_path=obs_db,
                output_dir=None,
                dry_run=True,
                require_corpus_quality=False,
            )

        figure_names = {o.get("figure_name") for o in captured_obs}
        assert {"blocked_fig", "warned_fig", "ok_fig"} <= figure_names, (
            f"Without gate, all figures should reach build_training_records, got: {figure_names}"
        )

    def test_legacy_figures_pass_through_gate(self, obs_db):
        """
        Figures with document_count=0 (legacy ingests, no --doc-title used) must
        not be blocked. Only preliminary figures with docs>0 are blocked.
        """
        from cli.export import export as run_export, build_training_records as _orig_build

        def legacy_quality(db_path, fig):
            return self._quality("preliminary", docs=0)

        captured_obs = []

        def capturing_build(obs, *a, **kw):
            captured_obs.extend(obs)
            return _orig_build(obs, *a, **kw)

        with patch("core.corpus.get_figure_corpus_quality", side_effect=legacy_quality), \
             patch("cli.export.build_training_records", side_effect=capturing_build):
            run_export(
                db_path=obs_db,
                output_dir=None,
                dry_run=True,
                require_corpus_quality=True,
                force=False,
            )

        figure_names = {o.get("figure_name") for o in captured_obs}
        assert len(figure_names) > 0, (
            "Legacy figures (docs=0) should not be blocked and should reach build_training_records"
        )

    def test_corpus_confidence_annotated_on_records(self, obs_db):
        from cli.export import export as run_export, build_training_records

        def confident_quality(db_path, fig):
            return self._quality("confident", docs=3)

        records = []
        original_build = build_training_records

        def capturing_build(obs, *a, **kw):
            recs = original_build(obs, *a, **kw)
            records.extend(recs)
            return recs

        with patch("core.corpus.get_figure_corpus_quality", side_effect=confident_quality), \
             patch("cli.export.build_training_records", side_effect=capturing_build):
            run_export(
                db_path=obs_db,
                output_dir=None,
                dry_run=True,
                require_corpus_quality=True,
            )

        assert len(records) > 0, (
            "Expected training records to verify corpus_confidence annotation"
        )
        assert all("corpus_confidence" in r for r in records), (
            f"Missing corpus_confidence in: {[r for r in records if 'corpus_confidence' not in r]}"
        )
        # Mock returns "confident" for every figure — annotation must propagate that value
        assert all(r["corpus_confidence"] == "confident" for r in records), (
            f"Expected all records to have corpus_confidence='confident', got: "
            f"{set(r['corpus_confidence'] for r in records)}"
        )


# ---------------------------------------------------------------------------
# Threshold constants sanity
# ---------------------------------------------------------------------------

class TestCorpusQualityConstants:
    def test_confident_threshold_greater_than_partial(self):
        assert CORPUS_MIN_DOCS_CONFIDENT > CORPUS_MIN_DOCS_PARTIAL

    def test_partial_threshold_at_least_two(self):
        assert CORPUS_MIN_DOCS_PARTIAL >= 2

    def test_type_threshold_at_least_two(self):
        assert CORPUS_MIN_TYPES_CONFIDENT >= 2

    def test_confident_requires_more_types_than_one(self):
        # The tier requires at least 2 types — single-type corpora can't be confident
        assert CORPUS_MIN_TYPES_CONFIDENT > 1

    def test_confident_docs_threshold_absolute_minimum(self):
        # Semantic guarantee: confident requires at least 3 documents
        assert CORPUS_MIN_DOCS_CONFIDENT >= 3

    def test_type_threshold_absolute_minimum(self):
        # Semantic guarantee: confident requires at least 2 distinct doc types
        assert CORPUS_MIN_TYPES_CONFIDENT >= 2
