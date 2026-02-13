from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from backend.dependencies import settings
from rag_kitcore.rag.pipelines.answer import answer_stream

router = APIRouter()

class Query(BaseModel):
    question: str

@router.post("/rag")
async def rag_answer(payload: Query):
    chunks = []
    async for chunk in answer_stream(payload.question, settings):
        chunks.append(chunk)
    return {"answer": "".join(chunks)}

@router.post("/rag/stream")
async def rag_answer_stream(payload: Query):

    async def streamer():
        async for chunk in answer_stream(payload.question, settings):
            yield chunk

    return StreamingResponse(streamer(), media_type="text/plain")
