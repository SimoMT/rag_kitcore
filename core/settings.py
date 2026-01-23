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


class PromptSection(BaseModel):
    system: str
    human: str

class PromptsConfig(BaseModel):
    extractor: PromptSection
    # qa: PromptSection
    # summarizer: PromptSection

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
    prompts: PromptsConfig

    model_config = SettingsConfigDict(env_file=".env")

    @classmethod
    def from_yaml(cls, path="config/config.yaml", prompts_path="config/prompts.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        with open(prompts_path, "r") as f:
            prompts = yaml.safe_load(f)

        data["prompts"] = prompts
        return cls(**data)
