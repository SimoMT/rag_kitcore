from .vllm import VLLMClient
from .ollama import OllamaClient

def create_llm(settings):
    backend = settings.llm.backend.lower()

    if backend == "vllm":
        return VLLMClient(settings)
    if backend == "ollama":
        return OllamaClient(settings)

    raise ValueError(f"Unsupported LLM backend: {backend}. Supported: vllm, ollama")
