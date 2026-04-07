# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing the 'SemanticMarkdownChunker' class.

This chunker parses markdown into structural blocks and creates semantically
meaningful chunks that preserve document context through heading breadcrumbs.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from graphrag_chunking.chunker import Chunker
from graphrag_chunking.text_chunk import TextChunk


class _BlockType(StrEnum):
    """Types of markdown structural blocks."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FENCED_CODE = "fenced_code"
    LIST = "list"
    BLOCKQUOTE = "blockquote"
    THEMATIC_BREAK = "thematic_break"


@dataclass
class _Block:
    """A structural block of markdown content."""

    type: _BlockType
    content: str
    level: int = 0  # heading level (1-6), 0 for non-headings


@dataclass
class _ProtectedGroup:
    """A group of blocks that must not be separated (e.g. table + caption)."""

    blocks: list[_Block] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n\n".join(b.content for b in self.blocks)


class SemanticMarkdownChunker(Chunker):
    """A chunker that splits markdown documents by semantic structure.

    Key features:
    - Parses markdown into structural blocks (headings, paragraphs, tables,
      code blocks, lists, blockquotes)
    - Preserves heading hierarchy as breadcrumb context in each chunk
    - Protects tables with their titles/captions from being split
    - Keeps code blocks and lists intact when possible
    - Respects maximum token size with intelligent fallback splitting
    """

    def __init__(
        self,
        size: int = 1200,
        overlap: int = 0,
        encode: Callable[[str], list[int]] | None = None,
        decode: Callable[[list[int]], str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a semantic markdown chunker instance."""
        self._size = size
        self._encode = encode
        self._decode = decode

    def chunk(
        self, text: str, transform: Callable[[str], str] | None = None
    ) -> list[TextChunk]:
        """Chunk markdown text into semantically meaningful pieces."""
        if not text or not text.strip():
            return []

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        blocks = self._parse_blocks(text)
        groups = self._build_protected_groups(blocks)
        chunk_texts = self._assemble_chunks(groups)

        results = []
        for index, chunk_text in enumerate(chunk_texts):
            final_text = transform(chunk_text) if transform else chunk_text
            result = TextChunk(
                original=chunk_text,
                text=final_text,
                index=index,
                start_char=0,
                end_char=max(0, len(chunk_text) - 1),
            )
            if self._encode:
                result.token_count = len(self._encode(result.text))
            results.append(result)
        return results

    # ── Token counting ───────────────────────────────────────────────

    def _token_count(self, text: str) -> int:
        """Count tokens using the encoder or a character-based estimate."""
        if self._encode:
            return len(self._encode(text))
        return len(text) // 4

    # ── Phase 1: Markdown parsing ────────────────────────────────────

    def _parse_blocks(self, text: str) -> list[_Block]:
        """Parse markdown text into a flat sequence of structural blocks."""
        lines = text.split("\n")
        blocks: list[_Block] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            if not line.strip():
                i += 1
                continue

            # Fenced code block (``` or ~~~)
            fence_match = re.match(r"^(`{3,}|~{3,})(.*)", line)
            if fence_match:
                i = self._parse_fenced_code(lines, i, fence_match, blocks)
                continue

            # ATX heading (# through ######)
            heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
            if heading_match:
                blocks.append(
                    _Block(
                        type=_BlockType.HEADING,
                        content=line.rstrip(),
                        level=len(heading_match.group(1)),
                    )
                )
                i += 1
                continue

            # Thematic break (---, ***, ___) — check before table to avoid ambiguity
            if re.match(r"^(\*{3,}|-{3,}|_{3,})\s*$", line):
                blocks.append(
                    _Block(type=_BlockType.THEMATIC_BREAK, content=line.rstrip())
                )
                i += 1
                continue

            # Table (header row + separator row)
            if self._is_table_start(lines, i):
                i = self._parse_table(lines, i, blocks)
                continue

            # List (ordered or unordered)
            if re.match(r"^\s*([-*+]|\d+\.)\s", line):
                i = self._parse_list(lines, i, blocks)
                continue

            # Blockquote
            if line.lstrip().startswith(">"):
                i = self._parse_blockquote(lines, i, blocks)
                continue

            # Paragraph (default)
            i = self._parse_paragraph(lines, i, blocks)

        return blocks

    def _parse_fenced_code(
        self,
        lines: list[str],
        start: int,
        fence_match: re.Match,
        blocks: list[_Block],
    ) -> int:
        """Parse a fenced code block. Returns next line index."""
        fence_char = fence_match.group(1)[0]
        fence_len = len(fence_match.group(1))
        code_lines = [lines[start]]
        closing = re.compile(rf"^{re.escape(fence_char)}{{{fence_len},}}\s*$")
        i = start + 1

        while i < len(lines):
            code_lines.append(lines[i])
            if closing.match(lines[i]):
                i += 1
                break
            i += 1

        blocks.append(
            _Block(type=_BlockType.FENCED_CODE, content="\n".join(code_lines))
        )
        return i

    def _parse_table(
        self, lines: list[str], start: int, blocks: list[_Block]
    ) -> int:
        """Parse a markdown table. Returns next line index."""
        table_lines = [lines[start]]
        i = start + 1
        while i < len(lines) and self._is_table_line(lines[i]):
            table_lines.append(lines[i])
            i += 1
        blocks.append(_Block(type=_BlockType.TABLE, content="\n".join(table_lines)))
        return i

    def _parse_list(
        self, lines: list[str], start: int, blocks: list[_Block]
    ) -> int:
        """Parse a markdown list. Returns next line index."""
        list_lines = [lines[start]]
        base_match = re.match(r"^(\s*)", lines[start])
        base_indent = len(base_match.group(1)) if base_match else 0
        i = start + 1

        while i < len(lines):
            line = lines[i]
            if re.match(r"^\s*([-*+]|\d+\.)\s", line):
                list_lines.append(line)
                i += 1
            elif not line.strip():
                # Blank line between list items
                if i + 1 < len(lines) and re.match(
                    r"^\s*([-*+]|\d+\.)\s", lines[i + 1]
                ):
                    list_lines.append(line)
                    i += 1
                else:
                    break
            elif len(line) - len(line.lstrip()) > base_indent:
                # Indented continuation
                list_lines.append(line)
                i += 1
            else:
                break

        blocks.append(_Block(type=_BlockType.LIST, content="\n".join(list_lines)))
        return i

    def _parse_blockquote(
        self, lines: list[str], start: int, blocks: list[_Block]
    ) -> int:
        """Parse a blockquote block. Returns next line index."""
        quote_lines = [lines[start]]
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if line.lstrip().startswith(">"):
                quote_lines.append(line)
                i += 1
            elif line.strip() and not re.match(r"^#{1,6}\s", line):
                # Lazy continuation line
                quote_lines.append(line)
                i += 1
            else:
                break
        blocks.append(
            _Block(type=_BlockType.BLOCKQUOTE, content="\n".join(quote_lines))
        )
        return i

    def _parse_paragraph(
        self, lines: list[str], start: int, blocks: list[_Block]
    ) -> int:
        """Parse a paragraph (default block type). Returns next line index."""
        para_lines = [lines[start]]
        i = start + 1
        while i < len(lines):
            line = lines[i]
            if (
                not line.strip()
                or re.match(r"^#{1,6}\s", line)
                or re.match(r"^(`{3,}|~{3,})", line)
                or re.match(r"^(\*{3,}|-{3,}|_{3,})\s*$", line)
                or self._is_table_start(lines, i)
                or re.match(r"^\s*([-*+]|\d+\.)\s", line)
                or line.lstrip().startswith(">")
            ):
                break
            para_lines.append(line)
            i += 1
        blocks.append(
            _Block(type=_BlockType.PARAGRAPH, content="\n".join(para_lines))
        )
        return i

    # ── Table detection helpers ──────────────────────────────────────

    @staticmethod
    def _is_table_line(line: str) -> bool:
        """Check if a line looks like part of a markdown table."""
        stripped = line.strip()
        return "|" in stripped and not stripped.startswith("```")

    @staticmethod
    def _is_table_separator(line: str) -> bool:
        """Check if a line is a table separator row (| --- | --- |)."""
        return bool(re.match(r"^\|?[\s:-]+(\|[\s:-]+)+\|?\s*$", line.strip()))

    def _is_table_start(self, lines: list[str], i: int) -> bool:
        """Check if position i starts a valid table (header + separator)."""
        return (
            i + 1 < len(lines)
            and self._is_table_line(lines[i])
            and self._is_table_separator(lines[i + 1])
        )

    # ── Phase 2: Protected groups ────────────────────────────────────

    def _build_protected_groups(
        self, blocks: list[_Block]
    ) -> list[_ProtectedGroup | _Block]:
        """Group blocks that must not be split apart.

        Protected groups:
        - Paragraph immediately before a table -> kept together as title+table
        - Code blocks, standalone tables, and lists -> individually protected
        """
        result: list[_ProtectedGroup | _Block] = []
        i = 0

        while i < len(blocks):
            block = blocks[i]

            # Paragraph followed by table -> group as caption + table
            if (
                block.type == _BlockType.PARAGRAPH
                and i + 1 < len(blocks)
                and blocks[i + 1].type == _BlockType.TABLE
            ):
                result.append(_ProtectedGroup(blocks=[block, blocks[i + 1]]))
                i += 2
                continue

            # Individually protected blocks
            if block.type in (
                _BlockType.FENCED_CODE,
                _BlockType.TABLE,
                _BlockType.LIST,
            ):
                result.append(_ProtectedGroup(blocks=[block]))
                i += 1
                continue

            result.append(block)
            i += 1

        return result

    # ── Phase 3: Chunk assembly ──────────────────────────────────────

    def _assemble_chunks(
        self, items: list[_ProtectedGroup | _Block]
    ) -> list[str]:
        """Assemble final chunks with heading breadcrumb context.

        The heading stack tracks the current position in the document hierarchy.
        When starting a new chunk, the full heading breadcrumb is prepended to
        provide context about where the content sits in the document structure.

        Within a continuing chunk, headings appear inline as part of the natural
        document flow.
        """
        chunks: list[str] = []
        heading_stack: list[tuple[int, str]] = []  # (level, heading_line)
        unconsumed_headings: list[str] = []  # headings not yet placed in a chunk
        parts: list[str] = []  # content parts of the current chunk
        needs_context = True  # next content needs full breadcrumb

        def breadcrumb() -> str:
            """Build compact heading hierarchy from the stack."""
            return "\n".join(h[1] for h in heading_stack) if heading_stack else ""

        def emit() -> None:
            """Emit current chunk and reset state."""
            nonlocal parts, needs_context
            if parts:
                chunks.append("\n\n".join(parts))
            parts = []
            needs_context = True

        def would_fit(new_parts: list[str]) -> bool:
            """Check if adding new_parts to the current chunk stays within size."""
            test_text = "\n\n".join(parts + new_parts)
            return self._token_count(test_text) <= self._size

        for item in items:
            # ── Heading: update hierarchy, defer placement ──
            if isinstance(item, _Block) and item.type == _BlockType.HEADING:
                level = item.level
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, item.content))
                unconsumed_headings.append(item.content)
                continue

            # ── Thematic break: force chunk boundary ──
            if isinstance(item, _Block) and item.type == _BlockType.THEMATIC_BREAK:
                emit()
                unconsumed_headings = []
                continue

            # ── Content block ──
            content = item.content if isinstance(item, _Block) else item.content

            # Build prefix: full breadcrumb for new chunks, unconsumed headings otherwise
            if needs_context:
                bc = breadcrumb()
                prefix = [bc] if bc else []
            else:
                prefix = list(unconsumed_headings)
            unconsumed_headings = []

            to_add = prefix + [content]

            # Case 1: fits in current chunk
            if would_fit(to_add):
                parts.extend(to_add)
                needs_context = False
                continue

            # Case 2: overflow -> emit current, try fresh chunk with breadcrumb
            emit()
            bc = breadcrumb()
            fresh = [bc, content] if bc else [content]

            if self._token_count("\n\n".join(fresh)) <= self._size:
                parts = fresh
                needs_context = False
                continue

            # Case 3: content too large even alone -> split it
            bc_tokens = self._token_count(bc) if bc else 0
            sep_tokens = self._token_count("\n\n") if bc else 0
            available = max(self._size - bc_tokens - sep_tokens, self._size // 4)
            for sub_chunk in self._split_oversized(content, available):
                chunks.append(f"{bc}\n\n{sub_chunk}" if bc else sub_chunk)

        emit()
        return chunks

    # ── Phase 4: Oversized block splitting ───────────────────────────

    def _split_oversized(self, text: str, max_tokens: int) -> list[str]:
        """Split oversized content while preserving structure where possible.

        Strategy (in order of preference):
        1. Split by paragraphs (double newline) and merge into fitting chunks
        2. Split by lines (single newline) and merge
        3. Fall back to token-based splitting
        """
        if max_tokens <= 0:
            max_tokens = self._size // 2

        if self._token_count(text) <= max_tokens:
            return [text]

        # Try paragraph-level splits
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            return self._merge_and_split(paragraphs, max_tokens, "\n\n")

        # Try line-level splits
        lines = text.split("\n")
        if len(lines) > 1:
            return self._merge_and_split(lines, max_tokens, "\n")

        # Token-level fallback
        return self._token_split(text, max_tokens)

    def _merge_and_split(
        self, parts: list[str], max_tokens: int, separator: str
    ) -> list[str]:
        """Merge parts into chunks that fit, recursively splitting oversized parts."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = self._token_count(part)
            sep_tokens = self._token_count(separator) if current else 0

            if current and (current_tokens + part_tokens + sep_tokens) > max_tokens:
                chunks.append(separator.join(current))
                current = []
                current_tokens = 0

            if part_tokens > max_tokens:
                if current:
                    chunks.append(separator.join(current))
                    current = []
                    current_tokens = 0
                # Recursively split the oversized part
                chunks.extend(self._split_oversized(part, max_tokens))
                continue

            current.append(part)
            current_tokens += part_tokens + (
                self._token_count(separator) if len(current) > 1 else 0
            )

        if current:
            chunks.append(separator.join(current))
        return chunks

    def _token_split(self, text: str, max_tokens: int) -> list[str]:
        """Last resort: split text by token boundaries."""
        if self._encode and self._decode:
            all_tokens = self._encode(text)
            return [
                self._decode(all_tokens[i : i + max_tokens])
                for i in range(0, len(all_tokens), max_tokens)
            ]
        # Character-based fallback when no tokenizer is available
        char_limit = max_tokens * 4
        return [text[i : i + char_limit] for i in range(0, len(text), char_limit)]
