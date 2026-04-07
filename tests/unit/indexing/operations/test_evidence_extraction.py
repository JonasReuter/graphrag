# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for evidence-enhanced graph extraction.

Validates that the GraphExtractor correctly parses evidence fields
(confidence, completeness, source_quote) from the extended tuple format,
and falls back gracefully to defaults with the original 4/5-field format.
"""

import pandas as pd
import pytest

from graphrag.index.operations.extract_graph.graph_extractor import (
    GraphExtractor,
    _empty_evidence_df,
)
from graphrag.index.operations.extract_graph.extract_graph import _collect_evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(response: str):
    """Build a minimal async mock LLM that returns a fixed response."""
    from unittest.mock import AsyncMock, MagicMock

    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response
    mock.completion_async = AsyncMock(return_value=mock_response)
    return mock


EVIDENCE_FORMAT_RESPONSE = """\
("entity"<|>COMPANY A<|>ORGANIZATION<|>A company that does things<|>0.95<|>complete<|>Company A was established in 2010)
##
("entity"<|>PERSON B<|>PERSON<|>A person<|>0.6<|>partial<|>Person B worked there)
##
("relationship"<|>COMPANY A<|>PERSON B<|>Person B works at Company A<|>7<|>0.85<|>complete<|>Person B worked at Company A since 2015)
<|COMPLETE|>"""

ORIGINAL_FORMAT_RESPONSE = """\
("entity"<|>COMPANY A<|>ORGANIZATION<|>A company that does things)
##
("relationship"<|>COMPANY A<|>PERSON B<|>Person B works at Company A<|>7)
<|COMPLETE|>"""


# ---------------------------------------------------------------------------
# Parser unit tests (no actual LLM call)
# ---------------------------------------------------------------------------

class TestGraphExtractorParser:
    """Tests for the _process_result parser with evidence-extended format."""

    def _make_extractor(self, model=None):
        if model is None:
            model = _make_mock_model("")
        return GraphExtractor(
            model=model,
            prompt="unused",
            max_gleanings=0,
        )

    def test_evidence_format_entity_fields(self):
        extractor = self._make_extractor()
        entities_df, rels_df, evidence_df = extractor._process_result(
            EVIDENCE_FORMAT_RESPONSE, "tu1", "<|>", "##", extract_evidence=True
        )

        assert len(entities_df) == 2
        assert len(evidence_df) == 3  # 2 entities + 1 relationship

        ev_entity = evidence_df[evidence_df["subject_type"] == "entity"]
        assert len(ev_entity) == 2

        company_ev = ev_entity[ev_entity["subject_id"] == "COMPANY A"].iloc[0]
        assert company_ev["extraction_confidence"] == pytest.approx(0.95)
        assert company_ev["completeness_status"] == "complete"
        assert company_ev["source_span"] == "Company A was established in 2010"
        assert company_ev["text_unit_id"] == "tu1"

        partial_ev = ev_entity[ev_entity["subject_id"] == "PERSON B"].iloc[0]
        assert partial_ev["completeness_status"] == "partial"
        assert partial_ev["extraction_confidence"] == pytest.approx(0.6)

    def test_evidence_format_relationship_fields(self):
        extractor = self._make_extractor()
        _, rels_df, evidence_df = extractor._process_result(
            EVIDENCE_FORMAT_RESPONSE, "tu1", "<|>", "##", extract_evidence=True
        )

        assert len(rels_df) == 1
        rel_ev = evidence_df[evidence_df["subject_type"] == "relationship"].iloc[0]
        assert rel_ev["subject_id"] == "COMPANY A::PERSON B"
        assert rel_ev["extraction_confidence"] == pytest.approx(0.85)
        assert rel_ev["completeness_status"] == "complete"

    def test_original_format_backward_compatible(self):
        """4/5-field original format should still parse correctly."""
        extractor = self._make_extractor()
        entities_df, rels_df, evidence_df = extractor._process_result(
            ORIGINAL_FORMAT_RESPONSE, "tu1", "<|>", "##", extract_evidence=False
        )

        assert len(entities_df) == 1
        assert len(rels_df) == 1
        # No evidence when extract_evidence=False
        assert len(evidence_df) == 0

    def test_original_format_with_extract_evidence_flag(self):
        """Original 4/5-field format falls back to defaults when extract_evidence=True."""
        extractor = self._make_extractor()
        entities_df, rels_df, evidence_df = extractor._process_result(
            ORIGINAL_FORMAT_RESPONSE, "tu2", "<|>", "##", extract_evidence=True
        )

        assert len(entities_df) == 1
        ev = evidence_df[evidence_df["subject_type"] == "entity"].iloc[0]
        assert ev["extraction_confidence"] == pytest.approx(0.5)  # default
        assert ev["completeness_status"] == "unknown"  # default fallback
        assert ev["source_span"] is None

    def test_confidence_clamped_to_0_1(self):
        """Out-of-range confidence values are clamped."""
        bad_response = '("entity"<|>FOO<|>THING<|>desc<|>1.5<|>complete<|>foo bar)\n<|COMPLETE|>'
        extractor = self._make_extractor()
        _, _, evidence_df = extractor._process_result(
            bad_response, "tu1", "<|>", "##", extract_evidence=True
        )
        assert evidence_df.iloc[0]["extraction_confidence"] == pytest.approx(1.0)

    def test_invalid_confidence_defaults_to_0_5(self):
        bad_response = '("entity"<|>FOO<|>THING<|>desc<|>not_a_number<|>complete<|>foo bar)\n<|COMPLETE|>'
        extractor = self._make_extractor()
        _, _, evidence_df = extractor._process_result(
            bad_response, "tu1", "<|>", "##", extract_evidence=True
        )
        assert evidence_df.iloc[0]["extraction_confidence"] == pytest.approx(0.5)

    def test_invalid_completeness_falls_back_to_unknown(self):
        bad_response = '("entity"<|>FOO<|>THING<|>desc<|>0.9<|>nonsense_value<|>foo bar)\n<|COMPLETE|>'
        extractor = self._make_extractor()
        _, _, evidence_df = extractor._process_result(
            bad_response, "tu1", "<|>", "##", extract_evidence=True
        )
        assert evidence_df.iloc[0]["completeness_status"] == "unknown"

    def test_empty_result_returns_empty_dfs(self):
        extractor = self._make_extractor()
        entities_df, rels_df, evidence_df = extractor._process_result(
            "<|COMPLETE|>", "tu1", "<|>", "##", extract_evidence=True
        )
        assert len(entities_df) == 0
        assert len(rels_df) == 0
        assert len(evidence_df) == 0


# ---------------------------------------------------------------------------
# _collect_evidence tests
# ---------------------------------------------------------------------------

class TestCollectEvidence:
    """Tests for the _collect_evidence aggregation helper."""

    def _make_evidence_df(self, subject_id: str, text_unit_id: str) -> pd.DataFrame:
        return pd.DataFrame([{
            "subject_type": "entity",
            "subject_id": subject_id,
            "text_unit_id": text_unit_id,
            "source_span": "some text",
            "extraction_confidence": 0.8,
            "completeness_status": "complete",
        }])

    def test_collects_all_records(self):
        df1 = self._make_evidence_df("A", "tu1")
        df2 = self._make_evidence_df("B", "tu2")
        result = _collect_evidence([df1, df2])
        assert len(result) == 2

    def test_empty_dfs_skipped(self):
        df1 = self._make_evidence_df("A", "tu1")
        empty = pd.DataFrame(columns=df1.columns)
        result = _collect_evidence([df1, empty])
        assert len(result) == 1

    def test_all_empty_returns_empty_df(self):
        result = _collect_evidence([])
        assert len(result) == 0
        assert "subject_type" in result.columns

    def test_records_not_merged(self):
        """Unlike entities, evidence records with same subject are NOT merged."""
        df1 = self._make_evidence_df("A", "tu1")
        df2 = self._make_evidence_df("A", "tu2")  # same entity, different text unit
        result = _collect_evidence([df1, df2])
        assert len(result) == 2  # both records kept
