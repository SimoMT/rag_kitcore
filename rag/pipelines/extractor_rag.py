from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from rag.prompts.extractor import build_prompt
from retrievers.hybrid import get_final_context


def build_rag_chain(llm, resources):
    """
    Full RAG pipeline:
    query → retrieval → context → prompt → llm → text
    """

    # 1. Retrieval step
    def retrieve(query):
        docs = get_final_context(resources, query)
        return docs

    # 2. Convert docs to a single context string
    def build_context(docs):
        return "\n\n".join(
            f"--- DOC {i+1} ---\n{d.page_content}"
            for i, d in enumerate(docs)
        )

    prompt = build_prompt()

    return (
        {
            "question": RunnablePassthrough(),
            "context": RunnablePassthrough()
            | RunnableLambda(retrieve)
            | RunnableLambda(build_context),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
