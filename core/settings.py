from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import yaml

class Settings(BaseSettings):
    # -----------------------------
    # RAG
    # -----------------------------
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)

    # -----------------------------
    # Paths
    # -----------------------------
    data_dir: str = Field(default="data/")
    bm25_index: str = Field(default="data/bm25_index")
    qdrant_url: str = Field(default="http://localhost:6333")

    # -----------------------------
    # Vectorstore
    # -----------------------------
    collection_name: str = Field(default="rag_collection")

    # -----------------------------
    # Models
    # -----------------------------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_provider: str = Field(default="ollama")
    llm_model: str = Field(default="llama3.2:1b")

    # -----------------------------
    # Config
    # -----------------------------
    model_config = SettingsConfigDict(env_file=".env")

    @classmethod
    def from_yaml(cls, path: str = "config/config.yaml"):
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)

        flat = {}
        for section, values in yaml_data.items():
            for key, value in values.items():
                flat[key] = value

        return cls(**flat)


settings = Settings.from_yaml()
