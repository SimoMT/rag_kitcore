from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384


class VectorDBConfig(BaseModel):
    type: str  # "faiss", "qdrant", "weaviate"
    url: str | None = None
    index_path: str | None = None


class AppConfig(BaseModel):
    embeddings: EmbeddingConfig
    vectordb: VectorDBConfig
