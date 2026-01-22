import numpy as np

from config import AppConfig
from logging import get_logger

logger = get_logger(__name__)


class BaseVectorDB:
    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        raise NotImplementedError


class InMemoryFaissLikeDB(BaseVectorDB):
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: list[np.ndarray] = []
        self.metadatas: list[dict] = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        self.vectors.extend(vectors)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        if not self.vectors:
            return []
        all_vecs = np.vstack(self.vectors)
        dists = np.linalg.norm(all_vecs - query_vector, axis=1)
        idxs = np.argsort(dists)[:k]
        return [self.metadatas[i] for i in idxs]


def get_vector_db(config: AppConfig) -> BaseVectorDB:
    if config.vectordb.type == "faiss":
        logger.info("Using in-memory FAISS-like vector DB")
        return InMemoryFaissLikeDB(dim=config.embeddings.dim)
    # Placeholders for qdrant/weaviate clients
    raise NotImplementedError(f"Vector DB {config.vectordb.type} not implemented yet")
