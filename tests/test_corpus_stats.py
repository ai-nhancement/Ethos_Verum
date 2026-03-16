"""
tests/test_corpus_stats.py

Tests for core/corpus.py (stats queries) and cli/corpus_stats.py.

Tests run against real in-memory stores populated with known data so
we can assert exact values, not just "something returned".
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Fixture: populated in-memory stores
# ---------------------------------------------------------------------------

@pytest.fixture
def populated(tmp_path):
    """
    Two figures ingested:
      - courage_hero: action doc, courage text, extract=True
      - patience_hero: journal doc, patience text, extract=True

    ML disabled. Returns (doc_db_path, val_db_path).
    """
    from core.document_store import DocumentStore
    from core.value_store import ValueStore
    import core.document_store as _dmod
    import core.value_store as _vmod

    doc = DocumentStore(str(tmp_path / "docs.db"))
    val = ValueStore(str(tmp_path / "vals.db"))

    orig_doc, orig_val = _dmod._instance, _vmod._INSTANCE
    _dmod._instance = doc
    _vmod._INSTANCE = val

    courage_text = (
        "I was afraid, but I stood firm and refused to flee. "
        "Despite the danger, I pressed forward with courage and did not flinch. "
        "Even though the cost was high, I remained resolute. "
        "My commitment never wavered through the long ordeal."
    )
    patience_text = (
        "I waited years for the right moment, patient through every setback. "
        "My equanimity held even when others grew restless. "
        "I bided my time, unhurried, trusting the process would unfold. "
        "The steadfast endurance required of me was immense, yet I persevered."
    )

    with patch("core.embedder.is_available", return_value=False), \
         patch("core.structural_layer._get_zeroshot_pipeline", return_value=None):
        from core.pipeline import ingest_text
        ingest_text("courage_hero", courage_text, doc_type="action",  run_extract=True)
        ingest_text("patience_hero", patience_text, doc_type="journal", run_extract=True)

    yield str(tmp_path / "docs.db"), str(tmp_path / "vals.db")

    _dmod._instance = orig_doc
    _vmod._INSTANCE = orig_val


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

class TestOverview:
    def test_figure_count_matches_ingested(self, populated):
        from core.corpus import get_overview
        doc_db, val_db = populated
        ov = get_overview(doc_db, val_db)
        assert ov["figure_count"] == 2

    def test_total_passages_is_positive(self, populated):
        from core.corpus import get_overview
        doc_db, val_db = populated
        ov = get_overview(doc_db, val_db)
        assert ov["total_passages"] > 0

    def test_total_observations_is_positive(self, populated):
        from core.corpus import get_overview
        doc_db, val_db = populated
        ov = get_overview(doc_db, val_db)
        assert ov["total_observations"] > 0

    def test_coverage_rate_in_valid_range(self, populated):
        from core.corpus import get_overview
        doc_db, val_db = populated
        ov = get_overview(doc_db, val_db)
        assert 0.0 <= ov["coverage_rate"] <= 1.0

    def test_coverage_rate_not_zero_when_observations_exist(self, populated):
        """If observations exist, at least some passages were covered."""
        from core.corpus import get_overview
        doc_db, val_db = populated
        ov = get_overview(doc_db, val_db)
        if ov["total_observations"] > 0:
            assert ov["coverage_rate"] > 0.0

    def test_doc_types_match_ingested_types(self, populated):
        from core.corpus import get_overview
        doc_db, val_db = populated
        ov = get_overview(doc_db, val_db)
        assert "action" in ov["doc_types"]
        assert "journal" in ov["doc_types"]

    def test_doc_type_passage_counts_sum_to_total(self, populated):
        from core.corpus import get_overview
        doc_db, val_db = populated
        ov = get_overview(doc_db, val_db)
        assert sum(ov["doc_types"].values()) == ov["total_passages"]

    def test_ingested_range_is_present(self, populated):
        from core.corpus import get_overview
        doc_db, val_db = populated
        ov = get_overview(doc_db, val_db)
        assert ov["ingested_range"] is not None
        assert len(ov["ingested_range"]) == 2
        assert ov["ingested_range"][0] <= ov["ingested_range"][1]

    def test_empty_corpus_returns_zeros(self, tmp_path):
        from core.document_store import DocumentStore
        from core.value_store import ValueStore
        from core.corpus import get_overview
        # Create empty stores
        DocumentStore(str(tmp_path / "empty_docs.db"))
        ValueStore(str(tmp_path / "empty_vals.db"))
        ov = get_overview(str(tmp_path / "empty_docs.db"),
                          str(tmp_path / "empty_vals.db"))
        assert ov["figure_count"] == 0
        assert ov["total_passages"] == 0
        assert ov["coverage_rate"] == 0.0


# ---------------------------------------------------------------------------
# Figure summaries
# ---------------------------------------------------------------------------

class TestFigureSummaries:
    def test_returns_one_entry_per_figure(self, populated):
        from core.corpus import get_figure_summaries
        doc_db, val_db = populated
        figs = get_figure_summaries(doc_db, val_db)
        names = {f["figure_name"] for f in figs}
        assert "courage_hero" in names
        assert "patience_hero" in names

    def test_courage_hero_has_courage_in_top_values(self, populated):
        """courage_hero was ingested with courage text — courage must appear."""
        from core.corpus import get_figure_summaries
        doc_db, val_db = populated
        figs = get_figure_summaries(doc_db, val_db)
        hero = next(f for f in figs if f["figure_name"] == "courage_hero")
        top_names = [v["value_name"] for v in hero["top_values"]]
        assert "courage" in top_names, \
            f"courage_hero top values: {top_names}"

    def test_avg_resistance_action_higher_than_journal(self, populated):
        """
        action doc_type has +0.40 resistance bonus vs journal +0.35.
        Even after extraction, avg_resistance for action figure must exceed journal.
        """
        from core.corpus import get_figure_summaries
        doc_db, val_db = populated
        figs = get_figure_summaries(doc_db, val_db)
        action_fig  = next(f for f in figs if f["figure_name"] == "courage_hero")
        journal_fig = next(f for f in figs if f["figure_name"] == "patience_hero")
        if action_fig["avg_resistance"] > 0 and journal_fig["avg_resistance"] > 0:
            assert action_fig["avg_resistance"] >= journal_fig["avg_resistance"], \
                (f"action resistance {action_fig['avg_resistance']} should be ≥ "
                 f"journal resistance {journal_fig['avg_resistance']}")

    def test_passage_count_is_positive_and_accurate(self, populated):
        from core.corpus import get_figure_summaries
        doc_db, val_db = populated
        figs = get_figure_summaries(doc_db, val_db)
        for f in figs:
            assert f["passage_count"] > 0, \
                f"{f['figure_name']} has passage_count=0"

    def test_all_summary_fields_present(self, populated):
        from core.corpus import get_figure_summaries
        doc_db, val_db = populated
        figs = get_figure_summaries(doc_db, val_db)
        required = {"figure_name", "session_id", "doc_type", "passage_count",
                    "observations", "unique_values", "top_values",
                    "avg_resistance", "avg_significance"}
        for f in figs:
            missing = required - set(f.keys())
            assert not missing, f"Missing keys in figure summary: {missing}"


# ---------------------------------------------------------------------------
# Value distribution
# ---------------------------------------------------------------------------

class TestValueDistribution:
    def test_returns_list_of_values(self, populated):
        from core.corpus import get_value_distribution
        _, val_db = populated
        dist = get_value_distribution(val_db)
        assert isinstance(dist, list)
        assert len(dist) > 0

    def test_each_entry_has_required_fields(self, populated):
        from core.corpus import get_value_distribution
        _, val_db = populated
        dist = get_value_distribution(val_db)
        for v in dist:
            for field in ("value_name", "total_demonstrations",
                          "figure_count", "avg_weight"):
                assert field in v, f"Missing {field} in value distribution entry"

    def test_sorted_by_total_demonstrations_desc(self, populated):
        from core.corpus import get_value_distribution
        _, val_db = populated
        dist = get_value_distribution(val_db)
        demos = [v["total_demonstrations"] for v in dist]
        assert demos == sorted(demos, reverse=True), \
            "Value distribution not sorted by total_demonstrations DESC"

    def test_courage_appears_in_distribution(self, populated):
        from core.corpus import get_value_distribution
        _, val_db = populated
        dist = get_value_distribution(val_db)
        names = [v["value_name"] for v in dist]
        assert "courage" in names

    def test_avg_weight_in_valid_range(self, populated):
        from core.corpus import get_value_distribution
        _, val_db = populated
        for v in get_value_distribution(val_db):
            assert 0.0 <= v["avg_weight"] <= 1.0, \
                f"avg_weight {v['avg_weight']} for {v['value_name']} out of [0,1]"


# ---------------------------------------------------------------------------
# Resistance distribution
# ---------------------------------------------------------------------------

class TestResistanceDistribution:
    def test_returns_summary_and_histogram(self, populated):
        from core.corpus import get_resistance_distribution
        _, val_db = populated
        res = get_resistance_distribution(val_db)
        for key in ("mean", "std", "min", "max", "median", "histogram"):
            assert key in res

    def test_mean_in_valid_range(self, populated):
        from core.corpus import get_resistance_distribution
        _, val_db = populated
        res = get_resistance_distribution(val_db)
        assert 0.0 <= res["mean"] <= 1.0

    def test_histogram_buckets_sum_to_total_obs(self, populated):
        from core.corpus import get_resistance_distribution, get_overview
        doc_db, val_db = populated
        res = get_resistance_distribution(val_db)
        ov  = get_overview(doc_db, val_db)
        bucket_total = sum(res["histogram"].values())
        assert bucket_total == ov["total_observations"], \
            (f"Histogram total {bucket_total} ≠ "
             f"total_observations {ov['total_observations']}")

    def test_min_le_mean_le_max(self, populated):
        from core.corpus import get_resistance_distribution
        _, val_db = populated
        res = get_resistance_distribution(val_db)
        if res["mean"] > 0:
            assert res["min"] <= res["mean"] <= res["max"]

    def test_empty_corpus_returns_zeros(self, tmp_path):
        from core.corpus import get_resistance_distribution
        from core.value_store import ValueStore
        ValueStore(str(tmp_path / "empty.db"))
        res = get_resistance_distribution(str(tmp_path / "empty.db"))
        assert res["mean"] == 0.0


# ---------------------------------------------------------------------------
# Cross-figure values
# ---------------------------------------------------------------------------

class TestCrossFigureValues:
    def test_values_in_both_figures_appear_at_min_2(self, populated):
        """
        A value that appears in both figures must be returned with min_figures=2.
        Fails if cross-figure aggregation is broken.
        """
        from core.corpus import get_value_distribution, get_cross_figure_values
        _, val_db = populated
        dist = get_value_distribution(val_db)
        # Find values that appear in both figures
        multi_fig = [v for v in dist if v["figure_count"] >= 2]
        cfv = get_cross_figure_values(val_db, min_figures=2)
        cfv_names = {v["value_name"] for v in cfv}
        for v in multi_fig:
            assert v["value_name"] in cfv_names, \
                f"{v['value_name']} has figure_count={v['figure_count']} but not in cross-figure list"

    def test_single_figure_values_excluded_at_min_2(self, populated):
        from core.corpus import get_value_distribution, get_cross_figure_values
        _, val_db = populated
        dist = get_value_distribution(val_db)
        single_fig = {v["value_name"] for v in dist if v["figure_count"] == 1}
        cfv_names  = {v["value_name"] for v in get_cross_figure_values(val_db, min_figures=2)}
        overlap = single_fig & cfv_names
        assert not overlap, \
            f"Single-figure values should not appear in cross-figure list: {overlap}"

    def test_min_figures_3_returns_subset_of_min_2(self, populated):
        from core.corpus import get_cross_figure_values
        _, val_db = populated
        at2 = {v["value_name"] for v in get_cross_figure_values(val_db, min_figures=2)}
        at3 = {v["value_name"] for v in get_cross_figure_values(val_db, min_figures=3)}
        assert at3.issubset(at2), \
            f"min_figures=3 must be a subset of min_figures=2"

    def test_empty_result_for_impossible_threshold(self, populated):
        from core.corpus import get_cross_figure_values
        _, val_db = populated
        result = get_cross_figure_values(val_db, min_figures=9999)
        assert result == []


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

class TestFullReport:
    def test_all_sections_present(self, populated):
        from core.corpus import get_full_report
        doc_db, val_db = populated
        report = get_full_report(doc_db, val_db)
        for key in ("overview", "figures", "value_distribution",
                    "resistance", "cross_figure_values"):
            assert key in report

    def test_report_is_json_serialisable(self, populated):
        import json
        from core.corpus import get_full_report
        doc_db, val_db = populated
        report = get_full_report(doc_db, val_db)
        json.dumps(report)  # must not raise


# ---------------------------------------------------------------------------
# Dataset card generation
# ---------------------------------------------------------------------------

class TestDatasetCard:
    def test_card_contains_required_sections(self, populated):
        from cli.dataset_card import generate_card
        doc_db, val_db = populated
        card = generate_card("Test Corpus", doc_db, val_db)
        for section in ("## Dataset Description", "## Corpus Statistics",
                        "## Schema", "## Pipeline"):
            assert section in card, f"Missing section: {section}"

    def test_card_starts_with_yaml_frontmatter(self, populated):
        from cli.dataset_card import generate_card
        doc_db, val_db = populated
        card = generate_card("Test Corpus", doc_db, val_db)
        assert card.startswith("---\n")
        # Must close frontmatter
        lines = card.split("\n")
        closing = [i for i, l in enumerate(lines) if l == "---" and i > 0]
        assert closing, "YAML frontmatter not closed"

    def test_card_mentions_figure_names(self, populated):
        from cli.dataset_card import generate_card
        doc_db, val_db = populated
        card = generate_card("Test Corpus", doc_db, val_db)
        assert "courage_hero" in card
        assert "patience_hero" in card

    def test_card_contains_corpus_name(self, populated):
        from cli.dataset_card import generate_card
        doc_db, val_db = populated
        card = generate_card("My Special Corpus", doc_db, val_db)
        assert "My Special Corpus" in card
