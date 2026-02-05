# ---------------------------------------------
# BM25 RETRIEVAL TEST
# ---------------------------------------------

from rag_kitcore.core.settings import Settings
from rag_kitcore.rag.indexing.bm25_builder import load_bm25_index
from rag_kitcore.core.retrieval.bm25 import BM25RetrieverWrapper

def test_bm25():
    # Load settings
    settings = Settings.from_yaml("config/config.yaml")

    # Load the pickled LangChain BM25 retriever
    bm25_raw = load_bm25_index(settings.paths.bm25_index)

    print("Loaded BM25 index type:", type(bm25_raw))

    # Wrap it with your unified retriever
    bm25 = BM25RetrieverWrapper(bm25_raw)

    # Run retrieval
    query = "neural networks"
    docs = bm25.retrieve(query, top_k=5)

    print(f"\nBM25 returned {len(docs)} documents\n")

    for i, d in enumerate(docs):
        print(f"--- Document {i+1} ---")
        print(d.page_content[:300])
        print()


if __name__ == "__main__":
    test_bm25()
