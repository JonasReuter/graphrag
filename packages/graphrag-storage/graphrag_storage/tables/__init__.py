# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Table provider module for GraphRAG storage."""

from .table import Table
from .table_provider import TableProvider
from .table_type import TableType

__all__ = ["Table", "TableProvider", "TableType"]
