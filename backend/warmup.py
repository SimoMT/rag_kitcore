from backend.dependencies import settings, retriever, llm
from rag_kitcore.core.retrieval.factory import create_retriever
from rag_kitcore.core.llm.factory import create_llm

async def warmup():
    global retriever, llm
    retriever = create_retriever(settings)
    llm = create_llm(settings)
    llm("warm up", system_prompt="warm up")
