from typing import AsyncGenerator
from rag_kitcore.core.settings import Settings
from rag_kitcore.core.retrieval.factory import create_retriever
from rag_kitcore.rag.prompting.pipeline import build_prompt
from rag_kitcore.core.llm.factory import create_llm


def answer(query: str, settings: Settings) -> str:
    """
    Full RAG pipeline:
    1. Retrieve documents
    2. Build prompt
    3. Call LLM backend
    4. Return answer text
    """

    # 1. Build retriever (BM25 + Dense + Reranker)
    retriever = create_retriever(settings)
    docs = retriever.retrieve(query, top_k=settings.retrieval.top_k_rerank)

    # 2. Build prompt using your template
    prompt = build_prompt(query, docs, settings)

    # 3. Create LLM backend (OpenAI, Azure, Ollama, etc.)
    llm = create_llm(settings)

    # 4. Generate answer
    response = llm(prompt)

    return response

async def answer_stream(query: str, settings: Settings) -> AsyncGenerator[str, None]:
    """
    Streaming RAG pipeline:
    - Retrieve documents
    - Build prompt
    - Stream LLM output chunk by chunk
    """    
    retriever = create_retriever(settings)
    docs = retriever.retrieve(query, top_k=settings.retrieval.top_k_rerank)

    prompt = build_prompt(query, docs, settings)
    llm = create_llm(settings)

    # llm.stream() is a *sync* generator → wrap it
    for chunk in llm.stream(prompt):
        yield chunk
