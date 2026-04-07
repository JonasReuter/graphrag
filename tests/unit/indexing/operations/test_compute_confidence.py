# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the multi-factor confidence scoring operation."""

import pandas as pd
import pytest

from graphrag.index.operations.compute_confidence.compute_confidence import (
    compute_entity_relationship_confidence,
    _compute_confidence,
    _compute_group_metrics,
)


def _make_entities(titles: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"title": t, "type": "ORGANIZATION", "description": f"desc {t}", "attributes": None}
        for t in titles
    ])


def _make_relationships(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"source": s, "target": t, "description": "rel", "weight": 1.0, "attributes": None}
        for s, t in pairs
    ])


def _make_evidence(
    subject_type: str,
    subject_id: str,
    text_unit_id: str,
    extraction_confidence: float = 0.8,
    completeness_status: str = "complete",
    verification_status: str = "unverified",
    time_scope: str | None = None,
) -> dict:
    return {
        "subject_type": subject_type,
        "subject_id": subject_id,
        "text_unit_id": text_unit_id,
        "source_span": "some text",
        "extraction_confidence": extraction_confidence,
        "completeness_status": completeness_status,
        "verification_status": verification_status,
        "time_scope": time_scope,
    }


class TestComputeEntityRelationshipConfidence:
    """Tests for the main confidence computation function."""

    def test_enriches_entity_attributes(self):
        entities = _make_entities(["ACME"])
        rels = _make_relationships([])
        evidence = pd.DataFrame([
            _make_evidence("entity", "ACME", "tu1", 0.9, "complete", "confirmed"),
        ])

        enriched_entities, _, _ = compute_entity_relationship_confidence(
            entities_df=entities,
            relationships_df=rels,
            evidence_df=evidence,
            text_units_df=None,
        )

        attrs = enriched_entities.iloc[0]["attributes"]
        assert attrs is not None
        assert "confidence" in attrs
        assert 0.0 <= attrs["confidence"] <= 1.0
        assert attrs["evidence_count"] == 1
        assert attrs["is_partial"] is False

    def test_marks_partial_when_evidence_partial(self):
        entities = _make_entities(["ACME"])
        evidence = pd.DataFrame([
            _make_evidence("entity", "ACME", "tu1", 0.5, "partial"),
        ])
        enriched, _, _ = compute_entity_relationship_confidence(
            entities_df=entities,
            relationships_df=_make_relationships([]),
            evidence_df=evidence,
            text_units_df=None,
        )
        assert enriched.iloc[0]["attributes"]["is_partial"] is True

    def test_enriches_relationship_attributes(self):
        entities = _make_entities(["A", "B"])
        rels = _make_relationships([("A", "B")])
        evidence = pd.DataFrame([
            _make_evidence("relationship", "A::B", "tu1", 0.7, "complete", "confirmed"),
        ])
        _, enriched_rels, _ = compute_entity_relationship_confidence(
            entities_df=entities,
            relationships_df=rels,
            evidence_df=evidence,
            text_units_df=None,
        )
        attrs = enriched_rels.iloc[0]["attributes"]
        assert "confidence" in attrs
        assert attrs["evidence_count"] == 1

    def test_empty_evidence_returns_unchanged(self):
        entities = _make_entities(["ACME"])
        rels = _make_relationships([])
        evidence = pd.DataFrame(columns=[
            "subject_type", "subject_id", "text_unit_id",
            "source_span", "extraction_confidence", "completeness_status"
        ])

        enriched, _, contradictions = compute_entity_relationship_confidence(
            entities_df=entities,
            relationships_df=rels,
            evidence_df=evidence,
            text_units_df=None,
        )
        # Without evidence, attributes stay None
        assert enriched.iloc[0]["attributes"] is None
        assert len(contradictions) == 0

    def test_multiple_evidence_sources_increase_source_count(self):
        entities = _make_entities(["ACME"])
        evidence = pd.DataFrame([
            _make_evidence("entity", "ACME", "tu1", 0.8, "complete", "confirmed"),
            _make_evidence("entity", "ACME", "tu2", 0.9, "complete", "confirmed"),
        ])
        text_units = pd.DataFrame([
            {"id": "tu1", "document_id": "doc1"},
            {"id": "tu2", "document_id": "doc2"},
        ])
        enriched, _, _ = compute_entity_relationship_confidence(
            entities_df=entities,
            relationships_df=_make_relationships([]),
            evidence_df=evidence,
            text_units_df=text_units,
        )
        attrs = enriched.iloc[0]["attributes"]
        assert attrs["source_count"] == 2
        assert attrs["evidence_count"] == 2


class TestComputeConfidenceFormula:
    """Tests for the confidence formula itself."""

    def test_all_weights_sum_to_1(self):
        weights = {"extraction": 0.3, "source_agreement": 0.3, "cross_doc": 0.25, "contradiction_penalty": 0.15}
        metrics = {
            "avg_extraction_confidence": 1.0,
            "source_agreement_score": 1.0,
            "cross_doc_score": 1.0,
            "contradiction_penalty": 0.0,
        }
        score = _compute_confidence(metrics, weights)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_contradiction_penalty_reduces_score(self):
        weights = {"extraction": 0.25, "source_agreement": 0.25, "cross_doc": 0.25, "contradiction_penalty": 0.25}
        no_penalty = _compute_confidence(
            {"avg_extraction_confidence": 0.8, "source_agreement_score": 0.8,
             "cross_doc_score": 0.5, "contradiction_penalty": 0.0},
            weights,
        )
        with_penalty = _compute_confidence(
            {"avg_extraction_confidence": 0.8, "source_agreement_score": 0.8,
             "cross_doc_score": 0.5, "contradiction_penalty": 1.0},
            weights,
        )
        assert with_penalty < no_penalty

    def test_score_clamped_to_0_1(self):
        weights = {"extraction": 1.0, "source_agreement": 0.0, "cross_doc": 0.0, "contradiction_penalty": 2.0}
        score = _compute_confidence(
            {"avg_extraction_confidence": 1.0, "source_agreement_score": 1.0,
             "cross_doc_score": 1.0, "contradiction_penalty": 1.0},
            weights,
        )
        assert 0.0 <= score <= 1.0
