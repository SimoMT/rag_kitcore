from abc import ABC, abstractmethod
from typing import Any, List

class BaseVectorStore(ABC):

    @abstractmethod
    def create(self, dim: int) -> None:
        ...

    @abstractmethod
    def populate(self, docs: List[Any], embedder: Any) -> None:
        ...

    @abstractmethod
    def as_retriever(self) -> Any:
        """Return a retriever compatible with the RAG pipeline."""
        ...
