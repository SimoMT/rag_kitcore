from typing import List
from rag_kitcore.core.retrieval.base import BaseRetriever
from rag_kitcore.core.types import Document

class DenseRetriever(BaseRetriever):
    def __init__(self, retriever):
        """
        retriever: LangChain retriever from Qdrant
        """
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        lc_docs = self.retriever._get_relevant_documents(query, run_manager=None)[:top_k]

        return [
            Document(
                page_content=d.page_content,
                metadata=d.metadata or {}
            )
            for d in lc_docs
        ]
