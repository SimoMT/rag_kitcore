def get_final_context(resources, query: str, top_n: int = 5):
    vector_store = resources["vector_store"]
    bm25 = resources["bm25"]
    reranker = resources["reranker"]

    # -----------------------------
    # 1. Vector search (Qdrant)
    # -----------------------------
    try:
        v_results = vector_store.similarity_search(query, k=25)
    except Exception:
        v_results = []

    # -----------------------------
    # 2. BM25 search
    # -----------------------------
    try:
        bm25.k = 25
        bm25_results = bm25.get_relevant_documents(query)
    except Exception:
        bm25_results = []

    # -----------------------------
    # 3. Weighted merge
    # -----------------------------
    # You can tune these later or move them to config
    w_bm25 = 0.4
    w_vector = 0.6

    scored_docs = {}

    # Score BM25 docs
    for doc in bm25_results:
        scored_docs.setdefault(doc.page_content, {"doc": doc, "score": 0})
        scored_docs[doc.page_content]["score"] += w_bm25

    # Score vector docs
    for doc in v_results:
        scored_docs.setdefault(doc.page_content, {"doc": doc, "score": 0})
        scored_docs[doc.page_content]["score"] += w_vector

    # Convert to list
    merged_docs = [v["doc"] for v in scored_docs.values()]

    if not merged_docs:
        return []

    # -----------------------------
    # 4. Rerank with CrossEncoder
    # -----------------------------
    pairs = [[query, d.page_content] for d in merged_docs]
    scores = reranker.predict(pairs)

    for d, s in zip(merged_docs, scores):
        d.metadata["rerank_score"] = float(s)

    reranked = sorted(
        merged_docs,
        key=lambda x: x.metadata["rerank_score"],
        reverse=True,
    )

    return reranked[:top_n]
