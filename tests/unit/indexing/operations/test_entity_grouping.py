# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for entity_grouping (Union-Find + cosine grouping)."""

import math

import pytest

from graphrag.index.operations.resolve_entities.entity_grouping import (
    _UnionFind,
    build_candidate_groups,
)


# ---------------------------------------------------------------------------
# _UnionFind tests
# ---------------------------------------------------------------------------


class TestUnionFind:
    def test_each_node_is_own_root_initially(self):
        uf = _UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_connects_two_nodes(self):
        uf = _UnionFind(4)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_transitive_closure(self):
        """A~B, B~C => A, B, C all in same component."""
        uf = _UnionFind(3)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(1) == uf.find(2)

    def test_independent_components(self):
        uf = _UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)

    def test_self_union_is_noop(self):
        uf = _UnionFind(3)
        uf.union(1, 1)
        assert uf.find(1) == 1

    def test_single_node(self):
        uf = _UnionFind(1)
        assert uf.find(0) == 0


# ---------------------------------------------------------------------------
# build_candidate_groups tests
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, idx: int) -> list[float]:
    """Return a unit vector with 1.0 at position idx."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


def _norm_vec(*components: float) -> list[float]:
    """Return a normalised vector from the given components."""
    mag = math.sqrt(sum(c * c for c in components))
    return [c / mag for c in components]


class TestBuildCandidateGroups:
    def test_empty_input_returns_no_groups(self):
        assert build_candidate_groups([], [], threshold=0.9) == []

    def test_single_entity_returns_no_groups(self):
        groups = build_candidate_groups(["A"], [_unit_vec(3, 0)], threshold=0.9)
        assert groups == []

    def test_identical_vectors_grouped(self):
        """Two entities with identical embeddings should be in the same group."""
        v = _unit_vec(4, 0)
        groups = build_candidate_groups(["A", "B"], [v, v], threshold=0.99)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_orthogonal_vectors_not_grouped(self):
        """Orthogonal vectors have cosine similarity 0, must not be grouped."""
        groups = build_candidate_groups(
            ["A", "B"],
            [_unit_vec(4, 0), _unit_vec(4, 1)],
            threshold=0.9,
        )
        assert groups == []

    def test_three_way_transitive_group(self):
        """If A~B and B~C (above threshold), all three should form one group."""
        # All three vectors close together
        v0 = _norm_vec(1.0, 0.01, 0.0)
        v1 = _norm_vec(1.0, 0.0, 0.01)
        v2 = _norm_vec(0.99, 0.01, 0.01)
        groups = build_candidate_groups(["A", "B", "C"], [v0, v1, v2], threshold=0.98)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_two_independent_groups(self):
        """Two separate clusters should produce two independent groups."""
        # Cluster 1: indices 0, 1 — similar
        # Cluster 2: indices 2, 3 — similar but far from cluster 1
        v0 = _norm_vec(1.0, 0.01)
        v1 = _norm_vec(1.0, 0.02)
        v2 = _norm_vec(0.01, 1.0)
        v3 = _norm_vec(0.02, 1.0)
        groups = build_candidate_groups(
            ["A", "B", "C", "D"], [v0, v1, v2, v3], threshold=0.99
        )
        assert len(groups) == 2
        group_sets = [set(g) for g in groups]
        assert {0, 1} in group_sets
        assert {2, 3} in group_sets

    def test_threshold_respected(self):
        """Entities below threshold must not be grouped."""
        # cosine similarity ≈ 0.5 (45-degree angle in 2D)
        v0 = _norm_vec(1.0, 0.0)
        v1 = _norm_vec(1.0, 1.0)
        groups = build_candidate_groups(["A", "B"], [v0, v1], threshold=0.99)
        assert groups == []

    def test_only_groups_with_two_or_more_returned(self):
        """Singleton components must be excluded from results."""
        # All vectors must have the same dimension (3)
        v0 = _unit_vec(3, 0)               # [1,0,0]
        v1 = _unit_vec(3, 1)               # [0,1,0] — orthogonal to v0
        v2 = _norm_vec(0.0, 1.0, 0.001)    # almost [0,1,0], close to v1 but not v0
        v3 = _norm_vec(0.0, 1.0, 0.002)    # almost [0,1,0], close to v1 and v2
        groups = build_candidate_groups(
            ["A", "B", "C", "D"], [v0, v1, v2, v3], threshold=0.999
        )
        # A is isolated; B, C, D should form a group (all near [0,1,0])
        assert all(0 not in g for g in groups)
        merged = {idx for g in groups for idx in g}
        assert {1, 2, 3}.issubset(merged)
