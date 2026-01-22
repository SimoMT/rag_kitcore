class DummyLLM:
    def generate(self, prompt: str) -> str:
        # Placeholder – integrate OpenAI, Azure, Ollama, etc.
        return f"[LLM placeholder] {prompt[:200]}"
