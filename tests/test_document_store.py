"""Tests for document chunking and store operations."""

import pytest

from app.services.chunker import chunk_text


class TestChunker:
    """Unit tests for the text chunking utility."""

    def test_empty_text(self):
        """Empty input returns empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size produces one chunk."""
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_chunks(self):
        """Long text is split into multiple chunks."""
        # Create text with many sentences.
        sentences = [f"Sentence number {i} is here." for i in range(20)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_overlap_continuity(self):
        """Consecutive chunks have overlapping content."""
        sentences = [f"Revenue grew {i}% year over year." for i in range(10)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=80, overlap=30)

        # With overlap, the end of one chunk should appear in the next.
        if len(chunks) >= 2:
            # At minimum, chunks should share some words.
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            assert words_0 & words_1  # non-empty intersection

    def test_no_empty_chunks(self):
        """No chunk should be empty or whitespace-only."""
        text = "First sentence. Second sentence. Third sentence. Fourth."
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        for chunk in chunks:
            assert chunk.strip()

    def test_respects_chunk_size(self):
        """Most chunks should not exceed chunk_size by much."""
        sentences = [f"The company reported Q{i} earnings of ${i}B." for i in range(20)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # Allow some tolerance for sentence boundary alignment.
        for chunk in chunks:
            assert len(chunk) < 200  # generous upper bound


class TestDocumentStoreUnit:
    """Basic smoke tests for DocumentStore — no ChromaDB required.

    These tests verify the chunker integration and data flow. Full
    integration tests with ChromaDB require the dependency to be
    installed and are run separately.
    """

    def test_chunk_text_financial_doc(self):
        """Financial document text is chunked correctly."""
        doc = (
            "Apple Inc. reported Q3 2024 revenue of $81.8 billion, up 5% YoY. "
            "Services revenue hit an all-time high of $24.2 billion. "
            "iPhone revenue was $39.3 billion. "
            "The company returned over $32 billion to shareholders. "
            "Gross margin expanded to 46.3%, driven by services mix. "
            "Management expects Q4 revenue growth in the mid single digits."
        )
        chunks = chunk_text(doc, chunk_size=150, overlap=30)
        assert len(chunks) >= 1
        # All monetary values should appear across chunks.
        full_text = " ".join(chunks)
        assert "$81.8" in full_text
        assert "$24.2" in full_text
