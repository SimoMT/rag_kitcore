from rag_kitcore.core.vectorstore.qdrant_store import QdrantStore
from rag_kitcore.core.exceptions import VectorStoreError
from rag_kitcore.core.settings import Settings

def create_vector_store(settings: Settings):
    backend = settings.vectorstore.backend

    if backend == "qdrant":
        return QdrantStore(
            url=settings.paths.qdrant_url,
            collection_name=settings.vectorstore.collection_name,
        )

    raise VectorStoreError(f"Unknown vector store backend: {backend}")