"""Text chunking utilities for document ingestion.

Splits long documents into overlapping chunks suitable for embedding
and storage in a vector database.  Tries to split on sentence
boundaries to preserve semantic coherence.
"""

from __future__ import annotations

import re

from app.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into overlapping chunks.

    The function first tries to break on sentence boundaries.  If a
    single sentence exceeds *chunk_size* characters it falls back to
    a hard character split.

    Args:
        text: The input document text.
        chunk_size: Target maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive
            chunks for context continuity.

    Returns:
        A list of text chunks.  Empty input returns an empty list.
    """
    text = text.strip()
    if not text:
        return []

    # Split into sentences (period / question-mark / exclamation followed by space).
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed the chunk size, flush.
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # Keep overlap: walk backwards until we have enough chars.
            overlap_parts: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_parts.insert(0, s)
                overlap_len += len(s)

            current_chunk = overlap_parts
            current_length = overlap_len

        # Handle single sentences longer than chunk_size.
        if sentence_len > chunk_size:
            # Hard-split the sentence.
            for i in range(0, sentence_len, chunk_size - overlap):
                chunks.append(sentence[i : i + chunk_size])
            current_chunk = []
            current_length = 0
            continue

        current_chunk.append(sentence)
        current_length += sentence_len

    # Flush remaining.
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(
        "Chunked text (%d chars) → %d chunks (size=%d, overlap=%d)",
        len(text),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks
