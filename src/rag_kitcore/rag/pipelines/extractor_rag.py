from rag_kitcore.rag.prompting.extractor import build_prompt


class RAGPipeline:
    """
    Backend-agnostic RAG pipeline.
    Expects:
        - retriever: object with .get_relevant_documents(query)
        - llm: object with .generate(prompt) and .stream(prompt)
        - prompt_fn: function(question, context) -> str
    """

    def __init__(self, retriever, llm, settings):
        self.retriever = retriever
        self.llm = llm
        self.prompt_fn = build_prompt(settings=settings)

    # -----------------------------
    # Retrieval
    # -----------------------------
    def retrieve(self, query):
        return self.retriever.get_relevant_documents(query)

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
    def format_prompt(self, question, context):
        return self.prompt_fn(question, context)

    # -----------------------------
    # Non-streaming generation
    # -----------------------------
    def run(self, question):
        docs = self.retrieve(question)
        context = self.build_context(docs)
        prompt = self.format_prompt(question, context)
        return self.llm.generate(prompt)

    # -----------------------------
    # Streaming generation
    # -----------------------------
    def stream(self, question):
        docs = self.retrieve(question)
        context = self.build_context(docs)
        prompt = self.format_prompt(question, context)

        for chunk in self.llm.stream(prompt):
            yield chunk
