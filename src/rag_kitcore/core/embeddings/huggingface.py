from typing import List, Sequence
from langchain_huggingface import HuggingFaceEmbeddings
from rag_kitcore.core.embeddings.base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"device": device},
        )

        # Precompute dimension by embedding a dummy string
        test_vec = self.model.embed_query("test")
        self._dimension = len(test_vec)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self.model.embed_documents(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)

    @property
    def dimension(self) -> int:
        return self._dimension
