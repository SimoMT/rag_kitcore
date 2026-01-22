from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import RAGPipeline


class QueryRequest(BaseModel):
    query: str


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    app.state.pipeline = RAGPipeline(retriever=None, llm=None)


@app.post("/query")
async def query_rag(request: QueryRequest):
    pipeline: RAGPipeline = app.state.pipeline
    return {"answer": pipeline.run(request.query)}
