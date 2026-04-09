# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for the resolve_entities operation."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from graphrag.index.operations.resolve_entities.resolve_entities import (
    _groups_from_pairs,
    _rewrite_entities,
    _rewrite_relationships,
)


# ---------------------------------------------------------------------------
# FakeTable (same pattern as test_finalize_graph.py)
# ---------------------------------------------------------------------------


class FakeTable:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = list(rows or [])
        self._index = 0
        self.written: list[dict[str, Any]] = []

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self._rows):
            raise StopAsyncIteration
        row = dict(self._rows[self._index])
        self._index += 1
        return row

    async def write(self, row: dict[str, Any]) -> None:
        self.written.append(dict(row))

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(title: str, entity_id: str | None = None, text_unit_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": entity_id or str(uuid4()),
        "human_readable_id": 0,
        "title": title,
        "type": "PERSON",
        "description": f"Description of {title}",
        "text_unit_ids": text_unit_ids or ["tu1"],
        "frequency": 1,
        "degree": 1,
    }


def _rel(source: str, target: str, weight: float = 1.0) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "human_readable_id": 0,
        "source": source,
        "target": target,
        "description": "",
        "weight": weight,
        "combined_degree": 2,
        "text_unit_ids": ["tu1"],
    }


# ---------------------------------------------------------------------------
# _groups_from_pairs
# ---------------------------------------------------------------------------


class TestGroupsFromPairs:
    def test_simple_pair(self):
        groups = _groups_from_pairs([(0, 1)], n=3)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_transitive_triple(self):
        """(0,1) and (1,2) → one group {0,1,2}."""
        groups = _groups_from_pairs([(0, 1), (1, 2)], n=3)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_two_independent_pairs(self):
        groups = _groups_from_pairs([(0, 1), (2, 3)], n=4)
        assert len(groups) == 2
        sets = [set(g) for g in groups]
        assert {0, 1} in sets
        assert {2, 3} in sets

    def test_empty_pairs_returns_no_groups(self):
        assert _groups_from_pairs([], n=5) == []

    def test_only_nodes_in_pairs_are_included(self):
        """Node 2 is not in any pair — should not appear in results."""
        groups = _groups_from_pairs([(0, 1)], n=3)
        assert all(2 not in g for g in groups)


# ---------------------------------------------------------------------------
# _rewrite_entities
# ---------------------------------------------------------------------------


class TestRewriteEntities:
    async def test_alias_is_removed(self):
        """Alias entity must not appear in output."""
        all_entities = [_entity("SEBASTIAN MUSTERMANN"), _entity("HERR MUSTERMANN")]
        merge_map = {"HERR MUSTERMANN": "SEBASTIAN MUSTERMANN"}
        table = FakeTable(all_entities)
        await _rewrite_entities(table, all_entities, merge_map)

        written_titles = [r["title"] for r in table.written]
        assert "HERR MUSTERMANN" not in written_titles
        assert "SEBASTIAN MUSTERMANN" in written_titles

    async def test_text_unit_ids_merged(self):
        """Canonical entity must accumulate text_unit_ids from alias."""
        e1 = _entity("SEBASTIAN MUSTERMANN", text_unit_ids=["tu1"])
        e2 = _entity("HERR MUSTERMANN", text_unit_ids=["tu2"])
        all_entities = [e1, e2]
        merge_map = {"HERR MUSTERMANN": "SEBASTIAN MUSTERMANN"}
        table = FakeTable(all_entities)
        await _rewrite_entities(table, all_entities, merge_map)

        canonical = next(r for r in table.written if r["title"] == "SEBASTIAN MUSTERMANN")
        assert set(canonical["text_unit_ids"]) == {"tu1", "tu2"}

    async def test_no_merge_map_passes_through(self):
        """Empty merge_map: all entities pass through unchanged."""
        entities = [_entity("A"), _entity("B"), _entity("C")]
        table = FakeTable(entities)
        await _rewrite_entities(table, entities, {})
        assert len(table.written) == 3

    async def test_human_readable_ids_are_sequential(self):
        entities = [_entity("A"), _entity("B")]
        table = FakeTable(entities)
        await _rewrite_entities(table, entities, {})
        ids = [r["human_readable_id"] for r in table.written]
        assert ids == list(range(len(table.written)))

    async def test_unique_ids_assigned(self):
        entities = [_entity("A"), _entity("B")]
        table = FakeTable(entities)
        await _rewrite_entities(table, entities, {})
        written_ids = [r["id"] for r in table.written]
        assert len(set(written_ids)) == len(written_ids)


# ---------------------------------------------------------------------------
# _rewrite_relationships
# ---------------------------------------------------------------------------


class TestRewriteRelationships:
    async def test_alias_source_rewritten(self):
        rels = [_rel("HERR MUSTERMANN", "XYZ")]
        merge_map = {"HERR MUSTERMANN": "SEBASTIAN MUSTERMANN"}
        table = FakeTable(rels)
        await _rewrite_relationships(table, merge_map)
        assert table.written[0]["source"] == "SEBASTIAN MUSTERMANN"

    async def test_alias_target_rewritten(self):
        rels = [_rel("XYZ", "HERR MUSTERMANN")]
        merge_map = {"HERR MUSTERMANN": "SEBASTIAN MUSTERMANN"}
        table = FakeTable(rels)
        await _rewrite_relationships(table, merge_map)
        assert table.written[0]["target"] == "SEBASTIAN MUSTERMANN"

    async def test_self_loop_after_merge_removed(self):
        """A → B where B merges into A becomes A → A (self-loop), must be dropped."""
        rels = [_rel("SEBASTIAN MUSTERMANN", "HERR MUSTERMANN")]
        merge_map = {"HERR MUSTERMANN": "SEBASTIAN MUSTERMANN"}
        table = FakeTable(rels)
        await _rewrite_relationships(table, merge_map)
        assert table.written == []

    async def test_duplicate_edges_deduplicated(self):
        """After rewrite, (A, B) appearing twice must be merged into one row."""
        rels = [
            _rel("A", "B", weight=1.0),
            _rel("ALIAS_A", "B", weight=2.0),
        ]
        merge_map = {"ALIAS_A": "A"}
        table = FakeTable(rels)
        await _rewrite_relationships(table, merge_map)
        assert len(table.written) == 1
        assert table.written[0]["weight"] == 3.0

    async def test_text_unit_ids_unioned_on_dedup(self):
        """Deduplicated relationships must have the union of text_unit_ids."""
        r1 = _rel("A", "B")
        r1["text_unit_ids"] = ["tu1"]
        r2 = _rel("ALIAS_A", "B")
        r2["text_unit_ids"] = ["tu2"]
        merge_map = {"ALIAS_A": "A"}
        table = FakeTable([r1, r2])
        await _rewrite_relationships(table, merge_map)
        assert set(table.written[0]["text_unit_ids"]) == {"tu1", "tu2"}

    async def test_no_merge_map_passes_through(self):
        rels = [_rel("A", "B"), _rel("C", "D")]
        table = FakeTable(rels)
        await _rewrite_relationships(table, {})
        assert len(table.written) == 2

    async def test_human_readable_ids_sequential(self):
        rels = [_rel("A", "B"), _rel("C", "D")]
        table = FakeTable(rels)
        await _rewrite_relationships(table, {})
        ids = [r["human_readable_id"] for r in table.written]
        assert ids == [0, 1]
