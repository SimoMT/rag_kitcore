from typing import List
from rag_kitcore.core.retrieval.base import BaseRetriever
from rag_kitcore.core.types import Document

class BM25RetrieverWrapper(BaseRetriever):
    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        self.retriever.k = top_k

        # Newer LangChain BM25 requires run_manager=None
        lc_docs = self.retriever._get_relevant_documents(
            query,
            run_manager=None
        )

        return [
            Document(
                page_content=d.page_content,
                metadata=d.metadata or {},
            )
            for d in lc_docs
        ]
