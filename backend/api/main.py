from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_kitcore.core.settings import Settings
from rag_kitcore.rag.pipelines.answer import answer_stream
from rag_kitcore.core.retrieval.factory import create_retriever
from rag_kitcore.core.llm.factory import create_llm

app = FastAPI()
settings = Settings.from_yaml()

@app.on_event("startup")
async def warmup():
    retriever = create_retriever(settings)
    llm = create_llm(settings)
    llm("warm up", system_prompt="warm up")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "status": "ok",
        "device": settings.backends.device,
        "embedding_model": settings.models.embedding_model,
        "reranker_model": settings.models.reranker_model,
        "qdrant_url": settings.paths.qdrant_url,
        "ollama_url": settings.backends.ollama.base_url,
        "vectorstore_backend": settings.vectorstore.backend,
        "collection_name": settings.vectorstore.collection_name,
    }


class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "backend running", "message": "Hello!"}

@app.post("/rag")
async def rag_answer(payload: Query):
    """
    Non-streaming RAG endpoint.
    Collects all chunks and returns a single string.
    """
    chunks = []
    async for chunk in answer_stream(payload.question, settings):
        chunks.append(chunk)
    return {"answer": "".join(chunks)}


@app.post("/rag/stream")
async def rag_answer_stream(payload: Query):
    """
    Streaming RAG endpoint.
    Returns chunks as a streaming HTTP response.
    """

    async def streamer():
        async for chunk in answer_stream(payload.question, settings):
            yield chunk

    return StreamingResponse(streamer(), media_type="text/plain")
