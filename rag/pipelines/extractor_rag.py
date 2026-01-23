from rag.prompts.extractor import build_prompt
from retrievers.hybrid import get_final_context


class RAGPipeline:
    """
    A simple, backend-agnostic RAG pipeline.
    llm must expose:
        - generate(prompt: str) -> str
        - stream(prompt: str) -> Iterator[str]
    """

    def __init__(self, llm, resources):
        self.llm = llm
        self.resources = resources
        self.prompt_fn = build_prompt()

    # -----------------------------
    # Retrieval
    # -----------------------------
    def retrieve(self, query):
        return get_final_context(self.resources, query)

    # -----------------------------
    # Build context string
    # -----------------------------
    def build_context(self, docs):
        return "\n\n".join(
            f"--- DOC {i+1} ---\n{d.page_content}"
            for i, d in enumerate(docs)
        )

    # -----------------------------
    # Build final prompt
    # -----------------------------
    def build_prompt(self, question, context):
        return self.prompt_fn(question, context)

    # -----------------------------
    # Non-streaming generation
    # -----------------------------
    def run(self, question):
        docs = self.retrieve(question)
        context = self.build_context(docs)
        prompt = self.build_prompt(question, context)
        return self.llm.generate(prompt)

    # -----------------------------
    # Streaming generation
    # -----------------------------
    def stream(self, question):
        docs = self.retrieve(question)
        context = self.build_context(docs)
        prompt = self.build_prompt(question, context)

        for chunk in self.llm.stream(prompt):
            yield chunk


def build_rag_chain(llm, resources):
    """
    Factory function to keep compatibility with existing code.
    """
    return RAGPipeline(llm, resources)
