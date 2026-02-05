from typing import List
from rag_kitcore.core.types import Document
from rag_kitcore.core.rerankers.base import BaseReranker


class HybridRetriever:
    """
    Combines vector search + BM25 + reranking into a unified retriever.
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        reranker: BaseReranker,
        k_vector: int = 25,
        k_bm25: int = 25,
        w_vector: float = 0.6,
        w_bm25: float = 0.4,
        top_n: int = 5,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.k_vector = k_vector
        self.k_bm25 = k_bm25
        self.w_vector = w_vector
        self.w_bm25 = w_bm25
        self.top_n = top_n

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 1. Vector search
        try:
            v_results = self.vector_retriever.get_relevant_documents(query)
        except Exception:
            v_results = []

        # 2. BM25 search
        try:
            self.bm25_retriever.retriever.k = self.k_bm25
            bm25_results = self.bm25_retriever.get_relevant_documents(query)
        except Exception:
            bm25_results = []

        # 3. Weighted merge
        scored = {}

        for doc in bm25_results:
            scored.setdefault(doc.page_content, {"doc": doc, "score": 0})
            scored[doc.page_content]["score"] += self.w_bm25

        for doc in v_results:
            scored.setdefault(doc.page_content, {"doc": doc, "score": 0})
            scored[doc.page_content]["score"] += self.w_vector

        merged_docs = [v["doc"] for v in scored.values()]
        if not merged_docs:
            return []

        # 4. Rerank
        reranked = self.reranker.rerank(query, merged_docs)

        return reranked[: self.top_n]
