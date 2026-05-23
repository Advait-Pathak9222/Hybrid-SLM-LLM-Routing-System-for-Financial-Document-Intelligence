"""API routes for document ingestion, search, and management."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.services.document_store import document_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

document_router = APIRouter(prefix="/documents", tags=["documents"])


# ── Request / Response schemas ──────────────────────────────────────


class IngestRequest(BaseModel):
    """Payload for inline document ingestion."""

    text: str = Field(..., min_length=1, description="Full document text")
    title: str = Field(default="", description="Human-readable document title")
    source: str = Field(default="", description="Document source / origin")


class IngestResponse(BaseModel):
    """Result of a document ingestion."""

    doc_id: str
    chunks_stored: int
    message: str


class SearchResult(BaseModel):
    """A single semantic search result."""

    text: str
    doc_id: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    """Wrapper for search results."""

    query: str
    results: list[SearchResult]
    total_results: int


class DocumentInfo(BaseModel):
    """Metadata for a stored document."""

    doc_id: str
    title: str
    source: str
    total_chunks: int


# ── Helpers ─────────────────────────────────────────────────────────


def _require_store():
    """Raise 503 if the document store is not initialised."""
    if document_store is None:
        raise HTTPException(
            status_code=503,
            detail="Document store is not initialised. Is RAG enabled?",
        )
    return document_store


# ── Endpoints ───────────────────────────────────────────────────────


@document_router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """Chunk, embed, and store a financial document.

    The document text is split into overlapping chunks, embedded using
    the default sentence-transformer model, and stored in ChromaDB for
    later retrieval.
    """
    store = _require_store()
    logger.info(
        "Ingesting document | title=%r  source=%r  text_len=%d",
        request.title,
        request.source,
        len(request.text),
    )

    result = store.ingest(
        text=request.text,
        title=request.title,
        source=request.source,
    )

    return IngestResponse(
        doc_id=result["doc_id"],
        chunks_stored=result["chunks_stored"],
        message=f"Document ingested successfully: {result['chunks_stored']} chunks stored.",
    )


@document_router.post("/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Query(default="", description="Document title"),
    source: str = Query(default="", description="Document source"),
) -> IngestResponse:
    """Upload a .txt or .pdf file for ingestion.

    PDF files are parsed with ``pypdf``; plain text files are read
    directly.
    """
    store = _require_store()

    filename = file.filename or "unknown"
    content = await file.read()

    if filename.lower().endswith(".pdf"):
        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="pypdf is required for PDF uploads. pip install pypdf",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to parse PDF: {exc}"
            )
    else:
        text = content.decode("utf-8", errors="replace")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    effective_title = title or filename
    result = store.ingest(text=text, title=effective_title, source=source or filename)

    logger.info(
        "Uploaded document | file=%s  title=%r  chunks=%d",
        filename,
        effective_title,
        result["chunks_stored"],
    )

    return IngestResponse(
        doc_id=result["doc_id"],
        chunks_stored=result["chunks_stored"],
        message=f"File '{filename}' ingested: {result['chunks_stored']} chunks stored.",
    )


@document_router.get("/search", response_model=SearchResponse)
async def search_documents(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=3, ge=1, le=20, description="Number of results"),
) -> SearchResponse:
    """Semantic search across all ingested financial documents."""
    store = _require_store()

    results = store.search(query=query, top_k=top_k)

    return SearchResponse(
        query=query,
        results=[
            SearchResult(
                text=r.text,
                doc_id=r.doc_id,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ],
        total_results=len(results),
    )


@document_router.get("", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    """List all ingested documents with metadata."""
    store = _require_store()

    docs = store.list_documents()
    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            title=d.get("title", ""),
            source=d.get("source", ""),
            total_chunks=d.get("total_chunks", 0),
        )
        for d in docs
    ]


@document_router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """Delete a document and all its chunks from the store."""
    store = _require_store()

    deleted = store.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    return {"doc_id": doc_id, "chunks_deleted": deleted, "message": "Document deleted."}
