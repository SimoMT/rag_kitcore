from typing import List
from rag_kitcore.core.types import Document
from rag_kitcore.core.retrieval.base import BaseRetriever
from rag_kitcore.core.rerankers.base import BaseReranker

class RerankedRetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, reranker: BaseReranker, top_n: int = 5):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.top_n = top_n

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        docs = self.base_retriever.retrieve(query, top_k)
        if not docs:
            return []
        reranked = self.reranker.rerank(query, docs)
        return reranked[: self.top_n]
