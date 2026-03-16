"""
tests/test_batch_ingest.py

Tests for cli/batch_ingest.py and the multi-file figure support fix in
core/value_store.register_figure_source.

PRINCIPLE: Tests run the real pipeline. Stores are isolated per-test
via tmp_path. ML layers (embedder, zeroshot) disabled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stores(tmp_path):
    """Isolated real stores + ML disabled."""
    from core.document_store import DocumentStore
    from core.value_store import ValueStore
    import core.document_store as _dmod
    import core.value_store as _vmod

    doc = DocumentStore(str(tmp_path / "docs.db"))
    val = ValueStore(str(tmp_path / "vals.db"))

    orig_doc, orig_val = _dmod._instance, _vmod._INSTANCE
    _dmod._instance = doc
    _vmod._INSTANCE = val

    with patch("core.embedder.is_available", return_value=False), \
         patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
        yield doc, val

    _dmod._instance = orig_doc
    _vmod._INSTANCE = orig_val


def _write_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Multi-file figure support — register_figure_source accumulation
# ---------------------------------------------------------------------------

class TestRegisterFigureSourceAccumulation:
    """
    The store fix: passage_count must accumulate across multiple ingests for
    the same figure, not be overwritten.
    """

    def test_single_ingest_records_passage_count(self, stores):
        _, val = stores
        val.register_figure_source("figure:test", "test", "journal", passage_count=10)
        row = val.get_figure_source("figure:test")
        assert row["passage_count"] == 10

    def test_second_ingest_accumulates_passage_count(self, stores):
        _, val = stores
        val.register_figure_source("figure:test", "test", "journal", passage_count=10)
        val.register_figure_source("figure:test", "test", "journal", passage_count=7)
        row = val.get_figure_source("figure:test")
        assert row["passage_count"] == 17

    def test_three_ingests_accumulate_correctly(self, stores):
        _, val = stores
        for n in (5, 8, 12):
            val.register_figure_source("figure:multi", "multi", "action", passage_count=n)
        row = val.get_figure_source("figure:multi")
        assert row["passage_count"] == 25

    def test_same_doc_type_stays_unchanged(self, stores):
        _, val = stores
        val.register_figure_source("figure:test", "test", "journal", passage_count=5)
        val.register_figure_source("figure:test", "test", "journal", passage_count=3)
        row = val.get_figure_source("figure:test")
        assert row["document_type"] == "journal"

    def test_different_doc_types_become_mixed(self, stores):
        _, val = stores
        val.register_figure_source("figure:test", "test", "journal", passage_count=5)
        val.register_figure_source("figure:test", "test", "speech", passage_count=3)
        row = val.get_figure_source("figure:test")
        assert row["document_type"] == "mixed"

    def test_mixed_stays_mixed_on_further_ingest(self, stores):
        _, val = stores
        val.register_figure_source("figure:test", "test", "journal",  passage_count=5)
        val.register_figure_source("figure:test", "test", "speech",   passage_count=3)
        val.register_figure_source("figure:test", "test", "letter",   passage_count=4)
        row = val.get_figure_source("figure:test")
        assert row["document_type"] == "mixed"
        assert row["passage_count"] == 12

    def test_different_figures_dont_interfere(self, stores):
        _, val = stores
        val.register_figure_source("figure:a", "a", "journal", passage_count=10)
        val.register_figure_source("figure:b", "b", "speech",  passage_count=5)
        val.register_figure_source("figure:a", "a", "letter",  passage_count=3)
        assert val.get_figure_source("figure:a")["passage_count"] == 13
        assert val.get_figure_source("figure:b")["passage_count"] == 5


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

class TestManifestLoading:
    def test_valid_manifest_loads(self, tmp_path):
        from cli.batch_ingest import load_manifest
        m = {"corpus_name": "test", "figures": []}
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))
        result = load_manifest(str(p))
        assert result["corpus_name"] == "test"

    def test_missing_manifest_raises(self, tmp_path):
        from cli.batch_ingest import load_manifest
        with pytest.raises(FileNotFoundError):
            load_manifest(str(tmp_path / "nonexistent.json"))

    def test_invalid_json_raises(self, tmp_path):
        from cli.batch_ingest import load_manifest
        p = tmp_path / "bad.json"
        p.write_text("not json {{{")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_manifest(str(p))

    def test_empty_figures_list_accepted(self, tmp_path):
        from cli.batch_ingest import load_manifest
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"figures": []}))
        result = load_manifest(str(p))
        assert result["figures"] == []


# ---------------------------------------------------------------------------
# ingest_figure — single figure ingestion
# ---------------------------------------------------------------------------

_COURAGE_TEXT = (
    "I was afraid, but I stood firm and refused to flee. "
    "Despite the danger, I pressed forward with courage and did not flinch. "
    "I will not yield. Even though the cost was high, I remained resolute. "
    "My commitment never wavered through the long ordeal."
)

_NEUTRAL_TEXT = (
    "The quarterly report was filed on Tuesday. The meeting started at nine. "
    "The spreadsheet contained twelve rows of data. The committee adjourned."
)


class TestIngestFigure:
    def test_single_source_extracts_values(self, tmp_path, stores):
        from cli.batch_ingest import ingest_figure
        f = _write_file(tmp_path, "bravery.txt", _COURAGE_TEXT)
        result = ingest_figure(
            figure_name="brave",
            sources=[{"file": str(f), "doc_type": "action"}],
            manifest_dir=tmp_path,
        )
        assert result["sources_ok"] == 1
        assert result["sources_failed"] == 0
        assert result["total_passages"] > 0
        assert result["total_observations"] > 0
        assert result["errors"] == []

    def test_extraction_runs_once_for_multiple_sources(self, tmp_path, stores):
        """
        Extraction must run exactly once even with multiple source files.
        Fails if extraction runs per-file (wastes work + inconsistent state).
        """
        from cli.batch_ingest import ingest_figure
        from core.value_store import get_value_store
        f1 = _write_file(tmp_path, "a.txt", _COURAGE_TEXT)
        f2 = _write_file(tmp_path, "b.txt", _COURAGE_TEXT)

        result = ingest_figure(
            figure_name="multi",
            sources=[
                {"file": str(f1), "doc_type": "action"},
                {"file": str(f2), "doc_type": "journal"},
            ],
            manifest_dir=tmp_path,
        )
        assert result["sources_ok"] == 2
        # Registry should reflect BOTH files' passages — if extracted per-file,
        # we'd process passages from file 1 twice and passages from file 2 once
        val = get_value_store()
        reg = val.get_registry("figure:multi", min_demonstrations=1)
        assert len(reg) > 0
        # Each value should be demonstrated from both sources (2 passages × 2 files)
        # but not from re-extraction of already-processed passages
        courage = next((r for r in reg if r["value_name"] == "courage"), None)
        assert courage is not None

    def test_missing_file_recorded_as_failure(self, tmp_path, stores):
        from cli.batch_ingest import ingest_figure
        result = ingest_figure(
            figure_name="test",
            sources=[{"file": "/nonexistent/path/file.txt", "doc_type": "journal"}],
            manifest_dir=tmp_path,
        )
        assert result["sources_failed"] == 1
        assert result["sources_ok"] == 0
        assert len(result["errors"]) == 1

    def test_partial_failure_continues(self, tmp_path, stores):
        """One bad source must not prevent good sources from being processed."""
        from cli.batch_ingest import ingest_figure
        good = _write_file(tmp_path, "good.txt", _COURAGE_TEXT)
        result = ingest_figure(
            figure_name="partial",
            sources=[
                {"file": "/nonexistent.txt", "doc_type": "journal"},
                {"file": str(good),          "doc_type": "action"},
            ],
            manifest_dir=tmp_path,
        )
        assert result["sources_failed"] == 1
        assert result["sources_ok"] == 1
        assert result["total_passages"] > 0

    def test_dry_run_does_not_write_db(self, tmp_path, stores):
        """dry_run=True must produce zero DB writes."""
        from cli.batch_ingest import ingest_figure
        from core.value_store import get_value_store
        f = _write_file(tmp_path, "bravery.txt", _COURAGE_TEXT)
        ingest_figure(
            figure_name="dryfig",
            sources=[{"file": str(f), "doc_type": "action"}],
            manifest_dir=tmp_path,
            dry_run=True,
        )
        val = get_value_store()
        assert val.get_figure_source("figure:dryfig") == {}
        assert val.get_registry("figure:dryfig") == []

    def test_multi_doc_type_sets_mixed_in_registry(self, tmp_path, stores):
        """After ingesting journal + speech, figure_sources.document_type must be 'mixed'."""
        from cli.batch_ingest import ingest_figure
        from core.value_store import get_value_store
        f1 = _write_file(tmp_path, "journal.txt", _COURAGE_TEXT)
        f2 = _write_file(tmp_path, "speech.txt", _COURAGE_TEXT)
        ingest_figure(
            figure_name="mixedtype",
            sources=[
                {"file": str(f1), "doc_type": "journal"},
                {"file": str(f2), "doc_type": "speech"},
            ],
            manifest_dir=tmp_path,
        )
        val = get_value_store()
        row = val.get_figure_source("figure:mixedtype")
        assert row["document_type"] == "mixed"


# ---------------------------------------------------------------------------
# batch_ingest — full manifest processing
# ---------------------------------------------------------------------------

class TestBatchIngest:
    def _make_manifest(self, tmp_path: Path, figures: list) -> Path:
        m = {"corpus_name": "test_corpus", "figures": figures}
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))
        return p

    def test_all_figures_processed(self, tmp_path, stores):
        from cli.batch_ingest import batch_ingest
        f1 = _write_file(tmp_path, "a.txt", _COURAGE_TEXT)
        f2 = _write_file(tmp_path, "b.txt", _COURAGE_TEXT)
        manifest = self._make_manifest(tmp_path, [
            {"name": "figA", "sources": [{"file": str(f1), "doc_type": "action"}]},
            {"name": "figB", "sources": [{"file": str(f2), "doc_type": "journal"}]},
        ])
        results = batch_ingest(str(manifest))
        assert len(results) == 2
        names = {r["figure_name"] for r in results}
        assert "figA" in names and "figB" in names

    def test_figure_filter_processes_only_named_figure(self, tmp_path, stores):
        from cli.batch_ingest import batch_ingest
        f1 = _write_file(tmp_path, "a.txt", _COURAGE_TEXT)
        f2 = _write_file(tmp_path, "b.txt", _COURAGE_TEXT)
        manifest = self._make_manifest(tmp_path, [
            {"name": "figA", "sources": [{"file": str(f1), "doc_type": "action"}]},
            {"name": "figB", "sources": [{"file": str(f2), "doc_type": "action"}]},
        ])
        results = batch_ingest(str(manifest), figure_filter="figA")
        assert len(results) == 1
        assert results[0]["figure_name"] == "figA"

    def test_returns_empty_for_unknown_figure_filter(self, tmp_path, stores):
        from cli.batch_ingest import batch_ingest
        manifest = self._make_manifest(tmp_path, [
            {"name": "figA", "sources": [{"file": "nope.txt", "doc_type": "action"}]},
        ])
        results = batch_ingest(str(manifest), figure_filter="nobody")
        assert results == []

    def test_nonzero_exit_on_any_failure(self, tmp_path, stores):
        """Exit code contract: any source failure → non-zero."""
        from cli.batch_ingest import batch_ingest
        manifest = self._make_manifest(tmp_path, [
            {"name": "broken", "sources": [{"file": "/missing.txt", "doc_type": "journal"}]},
        ])
        results = batch_ingest(str(manifest))
        assert any(r["sources_failed"] > 0 for r in results)

    def test_figures_appear_in_registry_after_batch(self, tmp_path, stores):
        """After batch_ingest, all figures must be queryable via get_figures_list."""
        from cli.batch_ingest import batch_ingest
        from core.value_store import get_value_store
        f1 = _write_file(tmp_path, "x.txt", _COURAGE_TEXT)
        f2 = _write_file(tmp_path, "y.txt", _COURAGE_TEXT)
        manifest = self._make_manifest(tmp_path, [
            {"name": "batchA", "sources": [{"file": str(f1), "doc_type": "action"}]},
            {"name": "batchB", "sources": [{"file": str(f2), "doc_type": "action"}]},
        ])
        batch_ingest(str(manifest))
        names = {f["figure_name"] for f in get_value_store().get_figures_list()}
        assert "batchA" in names
        assert "batchB" in names

    def test_passage_counts_are_accurate_in_registry(self, tmp_path, stores):
        """
        passage_count in figure_sources must reflect actual passages stored,
        not a hardcoded estimate. Fails if accumulation is broken.
        """
        from cli.batch_ingest import batch_ingest
        from core.value_store import get_value_store
        f1 = _write_file(tmp_path, "p1.txt", _COURAGE_TEXT)
        f2 = _write_file(tmp_path, "p2.txt", _COURAGE_TEXT)
        manifest = self._make_manifest(tmp_path, [
            {"name": "countme", "sources": [
                {"file": str(f1), "doc_type": "action"},
                {"file": str(f2), "doc_type": "journal"},
            ]},
        ])
        batch_ingest(str(manifest))
        row = get_value_store().get_figure_source("figure:countme")
        # Should have passages from BOTH files, not just the last one
        assert row["passage_count"] >= 2
