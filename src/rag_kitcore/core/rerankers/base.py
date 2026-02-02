from abc import ABC, abstractmethod
from typing import List, Any


class BaseReranker(ABC):

    @abstractmethod
    def rerank(self, query: str, docs: List[Any]) -> List[Any]:
        """Return documents sorted by relevance."""
        ...
