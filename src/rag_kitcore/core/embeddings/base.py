from abc import ABC, abstractmethod
from typing import List, Sequence


class BaseEmbedder(ABC):
    """Abstract interface for all embedding backends."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of documents. Returns a list of vectors."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string. Returns a single vector."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...
