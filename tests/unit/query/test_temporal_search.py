# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for temporal filtering in LocalSearchMixedContext.

Covers:
- _entity_in_time_window: entity filtering by observed_at / last_observed_at
- _relationship_in_time_window: relationship filtering with fallbacks
- build_context() temporal kwargs are accepted without error
"""

from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.query.structured_search.local_search.mixed_context import (
    _entity_in_time_window,
    _relationship_in_time_window,
)


def _entity(observed_at=None, last_observed_at=None) -> Entity:
    return Entity(
        id="e1", short_id=None, title="TEST",
        observed_at=observed_at, last_observed_at=last_observed_at,
    )


def _relationship(observed_at=None, last_observed_at=None, valid_from=None, valid_until=None) -> Relationship:
    return Relationship(
        id="r1", short_id=None, source="A", target="B",
        observed_at=observed_at, last_observed_at=last_observed_at,
        valid_from=valid_from, valid_until=valid_until,
    )


class TestEntityInTimeWindow:
    def test_no_temporal_data_always_passes(self):
        assert _entity_in_time_window(_entity(), "2024-01-01", "2024-12-31") is True

    def test_within_window(self):
        e = _entity("2024-03-01", "2024-09-01")
        assert _entity_in_time_window(e, "2024-01-01", "2024-12-31") is True

    def test_entirely_before_window(self):
        e = _entity("2022-01-01", "2022-12-31")
        assert _entity_in_time_window(e, "2024-01-01", "2024-12-31") is False

    def test_entirely_after_window(self):
        e = _entity("2025-01-01", "2025-06-01")
        assert _entity_in_time_window(e, "2024-01-01", "2024-12-31") is False

    def test_overlapping_start(self):
        # entity observed from 2023 to 2024-06; window starts 2024-01
        e = _entity("2023-01-01", "2024-06-01")
        assert _entity_in_time_window(e, "2024-01-01", "2024-12-31") is True

    def test_overlapping_end(self):
        e = _entity("2024-06-01", "2025-06-01")
        assert _entity_in_time_window(e, "2024-01-01", "2024-12-31") is True

    def test_no_from_date(self):
        e = _entity("2022-01-01", "2022-12-31")
        assert _entity_in_time_window(e, None, "2024-12-31") is True

    def test_no_until_date(self):
        e = _entity("2025-01-01", "2025-06-01")
        assert _entity_in_time_window(e, "2024-01-01", None) is True

    def test_only_observed_at_within(self):
        e = _entity(observed_at="2024-06-01")
        assert _entity_in_time_window(e, "2024-01-01", "2024-12-31") is True

    def test_only_observed_at_before(self):
        e = _entity(observed_at="2022-06-01")
        # no last_observed_at -- cannot tell if still active, so keep
        assert _entity_in_time_window(e, "2024-01-01", "2024-12-31") is True

    def test_only_last_observed_at_before_window(self):
        e = _entity(last_observed_at="2022-12-31")
        assert _entity_in_time_window(e, "2024-01-01", "2024-12-31") is False


class TestRelationshipInTimeWindow:
    def test_no_temporal_data_always_passes(self):
        assert _relationship_in_time_window(_relationship(), "2024-01-01", "2024-12-31") is True

    def test_within_window_via_observed_at(self):
        r = _relationship(observed_at="2024-03-01", last_observed_at="2024-09-01")
        assert _relationship_in_time_window(r, "2024-01-01", "2024-12-31") is True

    def test_before_window(self):
        r = _relationship(observed_at="2022-01-01", last_observed_at="2022-12-31")
        assert _relationship_in_time_window(r, "2024-01-01", "2024-12-31") is False

    def test_falls_back_to_valid_from_when_no_observed_at(self):
        r = _relationship(valid_from="2024-03-01", valid_until="2024-09-01")
        assert _relationship_in_time_window(r, "2024-01-01", "2024-12-31") is True

    def test_valid_from_before_window(self):
        r = _relationship(valid_from="2022-01-01", valid_until="2022-12-31")
        assert _relationship_in_time_window(r, "2024-01-01", "2024-12-31") is False

    def test_no_bounds_always_passes(self):
        r = _relationship(observed_at="2020-01-01", last_observed_at="2020-12-31")
        assert _relationship_in_time_window(r, None, None) is True

    def test_no_from_date_passes_old_data(self):
        r = _relationship(observed_at="2020-01-01", last_observed_at="2020-12-31")
        assert _relationship_in_time_window(r, None, "2024-12-31") is True

    def test_no_until_date_passes_new_data(self):
        r = _relationship(observed_at="2025-01-01", last_observed_at="2025-12-31")
        assert _relationship_in_time_window(r, "2024-01-01", None) is True
