from typing import List
from rag_kitcore.core.settings import Settings
from rag_kitcore.core.retrieval.factory import create_retriever
from rag_kitcore.core.types import Document


def run_retrieval(query: str, settings: Settings) -> List[Document]:
    """
    Runs the hybrid retrieval pipeline:
    - BM25 sparse retrieval
    - Dense vector retrieval
    - Fusion + reranking
    """
    retriever = create_retriever(settings)

    # top_k for the final reranked output
    top_k = settings.retrieval.top_k_rerank

    docs = retriever.retrieve(query, top_k=top_k)
    return docs

# from rag_kitcore.core.settings import Settings
# from rag_kitcore.rag.retrieval.pipeline import run_retrieval

# settings = Settings.from_yaml("config/config.yaml")
# docs = run_retrieval("What is a neural network?", settings)
