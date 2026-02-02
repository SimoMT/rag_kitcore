from rag_kitcore.core.vectorstore.qdrant_store import QdrantStore
from rag_kitcore.core.exceptions import VectorStoreError

def create_vector_store(config):
    backend = config.vectorstore.backend

    if backend == "qdrant":
        return QdrantStore(
            url=config.paths.qdrant_url,
            collection_name=config.vectorstore.collection_name,
        )

    raise VectorStoreError(f"Unknown vector store backend: {backend}")
