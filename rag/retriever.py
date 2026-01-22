import numpy as np

from vectorstore.embeddings import EmbeddingModel
from vectorstore.qdrant_store import BaseVectorDB


class Retriever:
    def __init__(self, embedder: EmbeddingModel, vectordb: BaseVectorDB):
        self.embedder = embedder
        self.vectordb = vectordb

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        q_vec = self.embedder.embed([query])
        return self.vectordb.search(np.array(q_vec)[0], k=k)
