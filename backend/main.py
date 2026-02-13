from fastapi import FastAPI
from contextlib import asynccontextmanager

from backend.warmup import warmup
from backend.routers import health, info, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    await warmup()
    yield

app = FastAPI(lifespan=lifespan)


app.include_router(health.router)
app.include_router(info.router)
app.include_router(rag.router)


@app.get("/")
def root():
    return {"status": "backend running", "message": "Hello!"}
