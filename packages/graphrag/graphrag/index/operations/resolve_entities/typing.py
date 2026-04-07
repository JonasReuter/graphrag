# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Type definitions for entity resolution."""

from dataclasses import dataclass, field


@dataclass
class MergeGroup:
    """A group of entity titles that should be merged into a single canonical entity."""

    canonical: str
    aliases: list[str] = field(default_factory=list)
