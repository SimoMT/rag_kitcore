# src/rag_kitcore/api/main.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_kitcore.core.settings import Settings
from rag_kitcore.rag.pipelines.answer import answer_stream

settings = Settings() # type: ignore[call-arg]
app = FastAPI()


class Query(BaseModel):
    question: str


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
