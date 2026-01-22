class RAGError(Exception):
    """Base exception for the RAG pipeline."""
    pass


# -----------------------------
# Indexing / Ingestion Errors
# -----------------------------

class IndexingError(RAGError):
    """Base class for indexing exceptions."""
    pass


class DocumentConversionError(IndexingError):
    """Raised when document conversion (PDF/DOCX → text) fails."""
    pass


class DocumentValidationError(IndexingError):
    """Raised when the input document is invalid or empty."""
    pass


class ChunkingError(IndexingError):
    """Raised when text chunking fails."""
    pass


# -----------------------------
# Vectorstore Errors
# -----------------------------

class VectorStoreError(RAGError):
    """Raised when vectorstore operations fail."""
    pass
