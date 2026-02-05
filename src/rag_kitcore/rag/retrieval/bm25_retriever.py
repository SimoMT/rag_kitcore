from typing import List
from langchain_community.retrievers import BM25Retriever
from rag_kitcore.core.types import Document



class BM25RetrieverWrapper:
    """
    Thin wrapper around LangChain's BM25Retriever to provide
    a consistent interface for the RAG pipeline.
    """

    def __init__(self, retriever: BM25Retriever):
        self.retriever = retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)  # type: ignore[attr-defined]
