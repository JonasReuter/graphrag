# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for SearchMethod enum — verifies CLI-facing values and completeness."""

import pytest

from graphrag.config.enums import SearchMethod


class TestSearchMethodValues:
    """Verify that every SearchMethod has the correct CLI string value."""

    def test_local_value(self):
        assert SearchMethod.LOCAL.value == "local"

    def test_global_value(self):
        assert SearchMethod.GLOBAL.value == "global"

    def test_drift_value(self):
        assert SearchMethod.DRIFT.value == "drift"

    def test_basic_value(self):
        assert SearchMethod.BASIC.value == "basic"

    def test_graph_local_value(self):
        assert SearchMethod.GRAPH_LOCAL.value == "graph-local"

    def test_graph_drift_value(self):
        assert SearchMethod.GRAPH_DRIFT.value == "graph-drift"

    def test_graph_global_value(self):
        assert SearchMethod.GRAPH_GLOBAL.value == "graph-global"


class TestSearchMethodStrRepr:
    """__str__ must return the CLI value (used in argparse / click)."""

    def test_str_graph_local(self):
        assert str(SearchMethod.GRAPH_LOCAL) == "graph-local"

    def test_str_graph_drift(self):
        assert str(SearchMethod.GRAPH_DRIFT) == "graph-drift"

    def test_str_graph_global(self):
        assert str(SearchMethod.GRAPH_GLOBAL) == "graph-global"

    def test_str_local(self):
        assert str(SearchMethod.LOCAL) == "local"

    def test_str_global(self):
        assert str(SearchMethod.GLOBAL) == "global"


class TestSearchMethodCompleteness:
    """All expected search modes must exist — catches accidental deletions."""

    EXPECTED_VALUES = {"local", "global", "drift", "basic", "graph-local", "graph-drift", "graph-global"}

    def test_no_missing_modes(self):
        actual = {m.value for m in SearchMethod}
        assert self.EXPECTED_VALUES == actual, (
            f"Missing: {self.EXPECTED_VALUES - actual}, "
            f"unexpected: {actual - self.EXPECTED_VALUES}"
        )

    def test_all_arangodb_modes_share_graph_prefix(self):
        arangodb_modes = [SearchMethod.GRAPH_LOCAL, SearchMethod.GRAPH_DRIFT, SearchMethod.GRAPH_GLOBAL]
        for mode in arangodb_modes:
            assert mode.value.startswith("graph-"), (
                f"{mode.name} should start with 'graph-', got '{mode.value}'"
            )


class TestSearchMethodLookup:
    """Enum must be constructable from its string value (for CLI parsing)."""

    @pytest.mark.parametrize("value,expected", [
        ("graph-local", SearchMethod.GRAPH_LOCAL),
        ("graph-drift", SearchMethod.GRAPH_DRIFT),
        ("graph-global", SearchMethod.GRAPH_GLOBAL),
        ("local", SearchMethod.LOCAL),
        ("global", SearchMethod.GLOBAL),
        ("drift", SearchMethod.DRIFT),
        ("basic", SearchMethod.BASIC),
    ])
    def test_lookup_by_value(self, value: str, expected: SearchMethod):
        assert SearchMethod(value) == expected

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError):
            SearchMethod("not-a-method")
