from rag_kitcore.core.settings import Settings
from rag_kitcore.core.retrieval.bm25 import BM25RetrieverWrapper
from rag_kitcore.core.retrieval.dense import DenseRetriever
from rag_kitcore.core.retrieval.hybrid import HybridRetriever
from rag_kitcore.core.embeddings.factory import create_embedder
from rag_kitcore.core.vectorstore.factory import create_vector_store
from rag_kitcore.core.rerankers.factory import create_reranker
from rag_kitcore.rag.indexing.bm25_builder import load_bm25_index


def create_retriever(settings: Settings):
    # --- Sparse (BM25) ---
    bm25_index = load_bm25_index(settings.paths.bm25_index)
    sparse = BM25RetrieverWrapper(bm25_index)

    # --- Dense (Qdrant) ---
    vector_store = create_vector_store(settings)
    embedder = create_embedder(settings)
    vector_store.load(embedder)
    dense = DenseRetriever(vector_store.as_retriever())

    # --- Reranker ---
    reranker = create_reranker(settings)

    # --- Hybrid ---
    return HybridRetriever(
        dense=dense,
        sparse=sparse,
        reranker=reranker,
        k_dense=settings.retrieval.top_k_dense,
        k_sparse=settings.retrieval.top_k_sparse,
        top_n=settings.retrieval.top_k_rerank,
    )
