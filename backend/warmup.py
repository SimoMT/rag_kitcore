from fastapi.concurrency import run_in_threadpool

from backend.dependencies import settings, retriever, llm
from rag_kitcore.core.retrieval.factory import create_retriever
from rag_kitcore.core.llm.factory import create_llm

async def warmup():
    global retriever, llm

    # Initialize components
    retriever = create_retriever(settings)
    llm = create_llm(settings)

    # Warm retriever (embedding model + vector DB)
    try:
        await run_in_threadpool(retriever.retrieve, "warm up")
    except Exception:
        pass

    # Warm LLM (load model into memory)
    try:
        await run_in_threadpool(llm, "warm up", system_prompt="warm up")
    except Exception:
        pass
