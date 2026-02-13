from typing import List, Dict
from rag_kitcore.core.retrieval.base import BaseRetriever
from rag_kitcore.core.types import Document
from rag_kitcore.core.rerankers.base import BaseReranker

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense: BaseRetriever,
        sparse: BaseRetriever,
        reranker: BaseReranker,
        k_dense: int = 25,
        k_sparse: int = 25,
        w_dense: float = 0.6,
        w_sparse: float = 0.4,
        top_n: int = 5,
    ):
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.w_dense = w_dense
        self.w_sparse = w_sparse
        self.top_n = top_n

    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        k_dense = top_k or self.k_dense
        k_sparse = top_k or self.k_sparse

        try:
            dense_docs = self.dense.retrieve(query, k_dense)
        except Exception:
            dense_docs = []

        try:
            sparse_docs = self.sparse.retrieve(query, k_sparse)
        except Exception:
            sparse_docs = []

        scored: Dict[str, dict] = {}

        for doc in sparse_docs:
            key = doc.page_content
            scored.setdefault(key, {"doc": doc, "score": 0.0})
            scored[key]["score"] += self.w_sparse

        for doc in dense_docs:
            key = doc.page_content
            scored.setdefault(key, {"doc": doc, "score": 0.0})
            scored[key]["score"] += self.w_dense

        merged_docs = [v["doc"] for v in scored.values()]
        if not merged_docs:
            return []

        reranked = self.reranker.rerank(query, merged_docs)
        return reranked[: (top_k or self.top_n)]
