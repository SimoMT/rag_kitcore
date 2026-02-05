# ---------------------------------------------
# HYBRID RETRIEVAL TEST (BM25 + DENSE + RERANKER)
# ---------------------------------------------

from rag_kitcore.core.settings import Settings
from rag_kitcore.core.retrieval.factory import create_retriever
from rag_kitcore.core.types import Document


def test_hybrid():
    # Load settings
    settings = Settings.from_yaml("config/config.yaml")

    # Build hybrid retriever (BM25 + Dense + Reranker)
    retriever = create_retriever(settings)

    # Run retrieval
    query = "refund policy"
    docs = retriever.retrieve(query, top_k=settings.retrieval.top_k_rerank)

    print(f"\nHybrid retriever returned {len(docs)} documents\n")

    for i, d in enumerate(docs):
        print(f"--- Document {i+1} ---")
        print(d.page_content[:300])
        print()


if __name__ == "__main__":
    test_hybrid()
