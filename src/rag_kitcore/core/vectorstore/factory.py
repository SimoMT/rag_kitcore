from rag_kitcore.core.vectorstore.qdrant_store import QdrantStore
from rag_kitcore.core.exceptions import VectorStoreError
from rag_kitcore.core.settings import Settings
from rag_kitcore.core.embeddings.factory import create_embedder

def create_vector_store(settings: Settings):
    backend = settings.vectorstore.backend

    if backend == "qdrant":
        store = QdrantStore(
            url=settings.paths.qdrant_url,
            collection_name=settings.vectorstore.collection_name,
        )

        embedder = create_embedder(settings)

        # Load existing collection
        store.load(embedder)

        return store
    
    raise VectorStoreError(f"Unknown vector store backend: {backend}")