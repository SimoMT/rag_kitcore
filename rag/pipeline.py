from .prompts import BASE_PROMPT
from .reranker import rerank


class RAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def run(self, query: str) -> str:
        docs = self.retriever.retrieve(query)
        docs = rerank(docs, query)
        context = "\n\n".join(d.get("text", "") for d in docs)
        prompt = BASE_PROMPT.format(context=context, question=query)
        return self.llm.generate(prompt)
