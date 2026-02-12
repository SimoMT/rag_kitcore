from typing import List
from rag_kitcore.core.types import Document  # or whatever your doc type is

class BaseRetriever:
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        raise NotImplementedError
