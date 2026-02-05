from core.settings import Settings
from rag.pipelines.extractor_rag import RAGPipeline
from core.llm_ollama import OllamaClient
from core.resources import load_resources

settings = Settings.from_yaml()

llm = OllamaClient(settings)
resources = load_resources(settings)

rag = RAGPipeline(llm, resources, settings)
