class Reranker:
    def __init__(self, model):
        self.model = model

    def rerank(self, query, docs):
        pairs = [[query, d.page_content] for d in docs]
        scores = self.model.predict(pairs)

        for d, s in zip(docs, scores):
            d.metadata["rerank_score"] = float(s)

        return sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
