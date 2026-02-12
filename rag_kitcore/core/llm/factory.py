from rag_kitcore.core.llm.ollama import OllamaClient

def create_llm(settings):
    backend = settings.llm.backend.lower()

    if backend == "ollama":
        return OllamaClient(settings)

    raise ValueError(f"Unsupported LLM backend: {backend}. Supported: vllm, ollama")
