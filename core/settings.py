from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


# -----------------------------
# Nested Config Models
# -----------------------------

class RagConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50


class PathsConfig(BaseModel):
    data_dir: str = "data/"
    bm25_index: str = "data/bm25_index"
    qdrant_url: str = "http://localhost:6333"


class VectorstoreConfig(BaseModel):
    collection_name: str = "rag_collection"


class ModelsConfig(BaseModel):
    embedding_model: str
    reranker_model: str


class LLMConfig(BaseModel):
    backend: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024


class BackendVLLM(BaseModel):
    base_url: str


class BackendOllama(BaseModel):
    base_url: str


class BackendsConfig(BaseModel):
    vllm: BackendVLLM
    ollama: BackendOllama


# -----------------------------
# Main Settings
# -----------------------------

class Settings(BaseSettings):
    rag: RagConfig
    paths: PathsConfig
    vectorstore: VectorstoreConfig
    models: ModelsConfig
    llm: LLMConfig
    backends: BackendsConfig

    model_config = SettingsConfigDict(env_file=".env")

    @classmethod
    def from_yaml(cls, path: str = "config/config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
