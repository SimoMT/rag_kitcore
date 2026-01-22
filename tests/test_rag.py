from rag.indexing.pipeline import RAGPipeline


class DummyRetriever:
    def retrieve(self, query: str):
        return [{"text": "dummy context"}]


class DummyLLM:
    def generate(self, prompt: str) -> str:
        return "dummy answer"


def test_rag_pipeline_runs():
    pipeline = RAGPipeline(retriever=DummyRetriever(), llm=DummyLLM())
    out = pipeline.run("test")
    assert "dummy" in out
