from typing import Iterable

from .embeddings import EmbeddingModel
from .vector_db import BaseVectorDB


def build_index(
    docs: Iterable[dict],
    embedder: EmbeddingModel,
    vectordb: BaseVectorDB,
    text_key: str = "text",
) -> None:
    texts = [d[text_key] for d in docs]
    vectors = embedder.embed(texts)
    vectordb.add(vectors, list(docs))
