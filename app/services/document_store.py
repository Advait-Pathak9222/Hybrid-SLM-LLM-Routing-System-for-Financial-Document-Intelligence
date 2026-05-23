"""ChromaDB-backed vector store for financial document retrieval.

Provides document ingestion (chunking + embedding + storage) and
semantic search so that the analysis pipeline can retrieve relevant
prior context before calling the language models.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.services.chunker import chunk_text
from app.utils.logger import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "financial_documents"


@dataclass(frozen=True)
class RetrievedChunk:
    """A single chunk returned from a semantic search."""

    text: str
    doc_id: str
    score: float
    metadata: dict


class DocumentStore:
    """Thin wrapper around a ChromaDB collection.

    Uses ChromaDB's default embedding function
    (``all-MiniLM-L6-v2`` via ``sentence-transformers``).

    Args:
        persist_dir: Filesystem path for ChromaDB's persistent storage.
    """

    def __init__(self, persist_dir: str = "./chroma_data") -> None:
        path = Path(persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "DocumentStore ready | persist=%s  docs=%d",
            persist_dir,
            self._collection.count(),
        )

    # ── Ingestion ───────────────────────────────────────────────

    def ingest(
        self,
        text: str,
        title: str = "",
        source: str = "",
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> dict:
        """Chunk, embed, and store a document.

        Args:
            text: Full document text.
            title: Human-readable title.
            source: Origin of the document (URL, filename, etc.).
            chunk_size: Maximum characters per chunk.
            overlap: Overlap between consecutive chunks.

        Returns:
            A dict with ``doc_id`` and ``chunks_stored``.
        """
        doc_id = uuid.uuid4().hex[:12]
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        if not chunks:
            logger.warning("No chunks produced for doc_id=%s, skipping", doc_id)
            return {"doc_id": doc_id, "chunks_stored": 0}

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        self._collection.add(documents=chunks, ids=ids, metadatas=metadatas)

        logger.info(
            "Ingested doc_id=%s  title=%r  chunks=%d", doc_id, title, len(chunks)
        )
        return {"doc_id": doc_id, "chunks_stored": len(chunks)}

    # ── Search ──────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Semantic search across all stored chunks.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.

        Returns:
            A list of :class:`RetrievedChunk` ordered by relevance.
        """
        if self._collection.count() == 0:
            logger.debug("Search skipped — collection is empty")
            return []

        results = self._collection.query(query_texts=[query], n_results=top_k)

        chunks: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    doc_id=meta.get("doc_id", ""),
                    score=round(1.0 - dist, 4),  # cosine distance → similarity
                    metadata=meta,
                )
            )

        logger.info("Search | query_len=%d  results=%d", len(query), len(chunks))
        return chunks

    # ── Management ──────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        """Return metadata for all unique documents in the store."""
        all_meta = self._collection.get().get("metadatas", [])
        seen: dict[str, dict] = {}
        for meta in all_meta:
            doc_id = meta.get("doc_id", "")
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "total_chunks": meta.get("total_chunks", 0),
                }
        return list(seen.values())

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to *doc_id*.

        Returns:
            The number of chunks deleted.
        """
        all_data = self._collection.get()
        ids_to_delete = [
            cid
            for cid, meta in zip(all_data["ids"], all_data["metadatas"])
            if meta.get("doc_id") == doc_id
        ]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        logger.info("Deleted doc_id=%s  chunks=%d", doc_id, len(ids_to_delete))
        return len(ids_to_delete)

    @property
    def total_chunks(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()


# Module-level singleton — lazily initialised by main.py lifespan.
document_store: DocumentStore | None = None


def init_document_store(persist_dir: str = "./chroma_data") -> DocumentStore:
    """Initialise (or re-initialise) the module-level document store."""
    global document_store  # noqa: PLW0603
    document_store = DocumentStore(persist_dir=persist_dir)
    return document_store
