from core.settings import Settings
from rag.pipelines.extractor_rag import RAGPipeline
from rag.llm.ollama import OllamaLLM
from retrievers.resources import load_resources

settings = Settings.from_yaml()

llm = OllamaLLM(
    host=settings.models.ollama_host,
    port=settings.models.ollama_port,
    model=settings.models.llm_model,
)

resources = load_resources(settings)

rag = RAGPipeline(llm, resources, settings)
