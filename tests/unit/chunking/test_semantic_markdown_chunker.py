# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the SemanticMarkdownChunker."""

from graphrag_chunking.chunk_strategy_type import ChunkerType
from graphrag_chunking.chunker_factory import create_chunker
from graphrag_chunking.chunking_config import ChunkingConfig
from graphrag_chunking.semantic_markdown_chunker import SemanticMarkdownChunker


def _char_encode(text: str) -> list[int]:
    """Simple encoder: 1 token per character (for predictable tests)."""
    return list(range(len(text)))


def _char_decode(tokens: list[int]) -> str:
    """Placeholder decode (not used in tests that don't need token splitting)."""
    return "x" * len(tokens)


def _make_chunker(size: int = 200) -> SemanticMarkdownChunker:
    """Create a chunker with character-based token counting."""
    return SemanticMarkdownChunker(size=size, encode=_char_encode, decode=_char_decode)


class TestEmptyAndTrivial:
    def test_empty_string(self):
        chunker = _make_chunker()
        assert chunker.chunk("") == []

    def test_whitespace_only(self):
        chunker = _make_chunker()
        assert chunker.chunk("   \n\n  ") == []

    def test_single_paragraph(self):
        chunker = _make_chunker()
        chunks = chunker.chunk("Hello world, this is a test.")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world, this is a test."
        assert chunks[0].index == 0

    def test_single_heading_with_paragraph(self):
        chunker = _make_chunker()
        text = "# Title\n\nSome content here."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert "# Title" in chunks[0].text
        assert "Some content here." in chunks[0].text


class TestHeadingBreadcrumbs:
    def test_heading_hierarchy_in_continuation(self):
        """When a chunk continues from a previous section, headings appear inline."""
        chunker = _make_chunker(size=500)
        text = "# Title\n\nIntro.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
        chunks = chunker.chunk(text)
        # With 500 char limit, everything should fit in one chunk
        assert len(chunks) == 1
        assert "# Title" in chunks[0].text
        assert "## Section A" in chunks[0].text
        assert "## Section B" in chunks[0].text

    def test_breadcrumb_on_overflow(self):
        """When content overflows to a new chunk, the heading breadcrumb is prepended."""
        # Each section has enough content to fill a chunk
        chunker = _make_chunker(size=80)
        text = (
            "# Doc\n\n"
            "## Section A\n\n"
            "This is content in section A that is fairly long.\n\n"
            "## Section B\n\n"
            "This is content in section B that is also long."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

        # Second chunk should have breadcrumb context
        last_chunk = chunks[-1]
        assert "# Doc" in last_chunk.text
        assert "## Section B" in last_chunk.text

    def test_deep_heading_hierarchy(self):
        """Deeply nested headings produce full breadcrumb trail."""
        chunker = _make_chunker(size=100)
        text = (
            "# Level 1\n\n"
            "## Level 2\n\n"
            "### Level 3\n\n"
            "Deep content here that is in the third level section."
        )
        chunks = chunker.chunk(text)
        # Find chunk with the deep content
        deep_chunk = [c for c in chunks if "Deep content" in c.text]
        assert len(deep_chunk) == 1
        assert "# Level 1" in deep_chunk[0].text
        assert "## Level 2" in deep_chunk[0].text
        assert "### Level 3" in deep_chunk[0].text

    def test_heading_level_change_updates_stack(self):
        """When heading level goes back up, the stack is correctly updated."""
        chunker = _make_chunker(size=60)
        text = (
            "# Root\n\n"
            "## A\n\n"
            "### A.1\n\n"
            "Content A.1 with some text.\n\n"
            "## B\n\n"
            "Content B with some text here."
        )
        chunks = chunker.chunk(text)
        # Find chunk with Content B
        b_chunk = [c for c in chunks if "Content B" in c.text]
        assert len(b_chunk) == 1
        assert "## B" in b_chunk[0].text
        # Should NOT contain ### A.1 (that was popped)
        assert "### A.1" not in b_chunk[0].text


class TestTableProtection:
    def test_table_with_caption_kept_together(self):
        """A paragraph followed by a table should stay in the same chunk."""
        chunker = _make_chunker(size=300)
        text = (
            "# Results\n\n"
            "The following table shows measurements:\n\n"
            "| Metric | Value |\n"
            "| --- | --- |\n"
            "| Accuracy | 95% |\n"
            "| Recall | 88% |"
        )
        chunks = chunker.chunk(text)
        # Table and its caption should be in the same chunk
        table_chunk = [c for c in chunks if "| Metric" in c.text]
        assert len(table_chunk) == 1
        assert "The following table shows measurements:" in table_chunk[0].text

    def test_standalone_table(self):
        """A table without a preceding paragraph is still protected."""
        chunker = _make_chunker(size=300)
        text = (
            "# Data\n\n"
            "| A | B | C |\n"
            "| --- | --- | --- |\n"
            "| 1 | 2 | 3 |\n"
            "| 4 | 5 | 6 |"
        )
        chunks = chunker.chunk(text)
        table_chunk = [c for c in chunks if "| A |" in c.text]
        assert len(table_chunk) == 1
        # All rows should be together
        assert "| 4 | 5 | 6 |" in table_chunk[0].text


class TestCodeBlockProtection:
    def test_fenced_code_block_kept_intact(self):
        """Fenced code blocks should not be split."""
        chunker = _make_chunker(size=300)
        text = (
            "# Example\n\n"
            "Here is some code:\n\n"
            "```python\n"
            "def hello():\n"
            '    print("Hello, world!")\n'
            "    return True\n"
            "```\n\n"
            "That was the code."
        )
        chunks = chunker.chunk(text)
        code_chunk = [c for c in chunks if "def hello" in c.text]
        assert len(code_chunk) == 1
        assert "```python" in code_chunk[0].text
        assert "return True" in code_chunk[0].text

    def test_tilde_code_block(self):
        """Tilde-fenced code blocks work the same."""
        chunker = _make_chunker(size=300)
        text = "~~~\ncode here\n~~~"
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert "code here" in chunks[0].text


class TestListProtection:
    def test_list_kept_together(self):
        """Lists should stay as one unit."""
        chunker = _make_chunker(size=300)
        text = (
            "# Items\n\n"
            "- First item\n"
            "- Second item\n"
            "- Third item\n"
            "- Fourth item"
        )
        chunks = chunker.chunk(text)
        list_chunk = [c for c in chunks if "First item" in c.text]
        assert len(list_chunk) == 1
        assert "Fourth item" in list_chunk[0].text

    def test_ordered_list(self):
        """Ordered lists are also protected."""
        chunker = _make_chunker(size=300)
        text = "1. Alpha\n2. Beta\n3. Gamma"
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert "3. Gamma" in chunks[0].text


class TestMaximumLength:
    def test_long_section_is_split(self):
        """Sections exceeding max size are split into multiple chunks."""
        chunker = _make_chunker(size=60)
        text = (
            "# Title\n\n"
            "This is a paragraph that is long enough to exceed the size limit. "
            "It contains multiple sentences for testing purposes."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # All chunks should have heading context
        for c in chunks:
            assert "# Title" in c.text

    def test_oversized_code_block_is_split(self):
        """Very large code blocks fall back to splitting."""
        chunker = _make_chunker(size=80)
        code_lines = "\n".join(f"line_{i} = {i}" for i in range(20))
        text = f"# Code\n\n```python\n{code_lines}\n```"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_token_count_on_chunks(self):
        """Each chunk should have a valid token_count when encoder is provided."""
        chunker = _make_chunker(size=100)
        text = "# Heading\n\nParagraph content.\n\n## Sub\n\nMore content here."
        chunks = chunker.chunk(text)
        for c in chunks:
            assert c.token_count is not None
            assert c.token_count > 0


class TestThematicBreaks:
    def test_thematic_break_forces_boundary(self):
        """A thematic break (---) forces a chunk boundary."""
        chunker = _make_chunker(size=500)
        text = "Part one.\n\n---\n\nPart two."
        chunks = chunker.chunk(text)
        assert len(chunks) == 2
        assert "Part one." in chunks[0].text
        assert "Part two." in chunks[1].text


class TestBlockquotes:
    def test_blockquote_parsed(self):
        """Blockquotes are correctly parsed and chunked."""
        chunker = _make_chunker(size=300)
        text = "# Quote\n\n> This is a blockquote.\n> It has two lines."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert "> This is a blockquote." in chunks[0].text
        assert "> It has two lines." in chunks[0].text


class TestTransform:
    def test_transform_applied(self):
        """The transform function is applied to chunk text."""
        chunker = _make_chunker()
        text = "# Title\n\nContent."
        chunks = chunker.chunk(text, transform=lambda t: f"PREFIX: {t}")
        assert chunks[0].text.startswith("PREFIX:")
        # Original should NOT have the prefix
        assert not chunks[0].original.startswith("PREFIX:")


class TestFactoryIntegration:
    def test_create_via_factory(self):
        """SemanticMarkdownChunker can be created through the factory."""
        config = ChunkingConfig(type=ChunkerType.SemanticMarkdown, size=200)
        chunker = create_chunker(config, encode=_char_encode, decode=_char_decode)
        assert isinstance(chunker, SemanticMarkdownChunker)
        chunks = chunker.chunk("# Hello\n\nWorld.")
        assert len(chunks) == 1


class TestWindowsLineEndings:
    def test_crlf_handling(self):
        """Windows line endings are normalized."""
        chunker = _make_chunker(size=300)
        text = "# Title\r\n\r\nContent.\r\n\r\n## Section\r\n\r\nMore."
        chunks = chunker.chunk(text)
        assert any("# Title" in c.text for c in chunks)
        assert any("More." in c.text for c in chunks)


class TestComplexDocument:
    def test_realistic_markdown(self):
        """Test with a realistic markdown document structure."""
        chunker = _make_chunker(size=250)
        text = """# Project Documentation

## Overview

This project implements a data processing pipeline
that handles multiple input formats.

## Architecture

### Data Layer

The data layer uses PostgreSQL for persistence.

#### Tables

The following tables exist:

| Table | Purpose |
| --- | --- |
| users | User accounts |
| events | Event log |

### API Layer

```python
@app.route("/api/data")
def get_data():
    return jsonify(data)
```

## Configuration

- `DB_HOST`: Database host
- `DB_PORT`: Database port
- `DB_NAME`: Database name

---

## Appendix

Additional notes go here."""

        chunks = chunker.chunk(text)

        # Should produce multiple chunks
        assert len(chunks) >= 2

        # Every chunk should have some content
        for c in chunks:
            assert len(c.text.strip()) > 0
            assert c.index >= 0
            assert c.token_count is not None

        # Table should be with its title
        table_chunks = [c for c in chunks if "| Table |" in c.text]
        assert len(table_chunks) == 1
        assert "The following tables exist:" in table_chunks[0].text

        # Code block should be intact
        code_chunks = [c for c in chunks if "get_data" in c.text]
        assert len(code_chunks) == 1
        assert "```python" in code_chunks[0].text

        # Thematic break should separate Appendix
        appendix_chunks = [c for c in chunks if "Additional notes" in c.text]
        assert len(appendix_chunks) == 1
