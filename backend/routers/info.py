from fastapi import APIRouter
from backend.dependencies import settings

router = APIRouter()

@router.get("/info")
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
