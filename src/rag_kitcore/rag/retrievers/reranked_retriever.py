from typing import List
from langchain_core.documents import Document
from rag_kitcore.core.rerankers.base import BaseReranker


class RerankedRetriever:
    """
    Wraps any retriever and applies a reranker on top.
    """

    def __init__(self, base_retriever, reranker: BaseReranker, top_n: int = 5):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.top_n = top_n

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        if not docs:
            return []

        reranked = self.reranker.rerank(query, docs)
        return reranked[: self.top_n]
