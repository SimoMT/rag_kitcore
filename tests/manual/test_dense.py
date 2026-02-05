# ---------------------------------------------
# DENSE RETRIEVAL TEST (QDRANT)
# ---------------------------------------------

from rag_kitcore.core.settings import Settings
from rag_kitcore.core.vectorstore.factory import create_vector_store
from rag_kitcore.core.retrieval.dense import DenseRetriever


def test_dense():
    # Load settings
    settings = Settings.from_yaml("config/config.yaml")

    # Load Qdrant store (existing collection)
    vector_store = create_vector_store(settings)

    # Wrap it into your unified retriever
    dense = DenseRetriever(vector_store.as_retriever())

    # Run retrieval
    query = "refund policy"
    docs = dense.retrieve(query, top_k=5)

    print(f"\nDense retriever returned {len(docs)} documents\n")

    for i, d in enumerate(docs):
        print(f"--- Document {i+1} ---")
        print(d.page_content[:300])
        print()


if __name__ == "__main__":
    test_dense()
