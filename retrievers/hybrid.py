from langchain_classic.retrievers import EnsembleRetriever


def get_final_context(resources, query: str, top_n: int = 5):
    vector_store = resources["vector_store"]
    bm25 = resources["bm25"]
    reranker = resources["reranker"]

    # Qdrant retriever (Runnable)
    v_retriever = vector_store.as_retriever(search_kwargs={"k": 25})

    # BM25 retriever (Runnable)
    bm25.k = 25

    ensemble = EnsembleRetriever(
        retrievers=[bm25, v_retriever],
        weights=[0.4, 0.6],
    )

    try:
        initial_docs = ensemble.invoke(query)
    except Exception:
        return []

    if not initial_docs:
        return []

    # Deduplicate
    unique_docs = {doc.page_content: doc for doc in initial_docs}
    docs = list(unique_docs.values())

    # Rerank
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    for d, s in zip(docs, scores):
        d.metadata["rerank_score"] = float(s)

    reranked = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
    return reranked[:top_n]
