from typing import List
from langchain_core.documents import Document


class VectorRetrieverWrapper:
    """
    Wraps a vector store retriever to provide a unified interface.
    """

    def __init__(self, retriever):
        """
        retriever must expose:
            get_relevant_documents(query: str) -> List[Document]
        """
        self.retriever = retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)
