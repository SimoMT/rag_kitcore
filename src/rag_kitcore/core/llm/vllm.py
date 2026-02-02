# core/llm/vllm.py

from typing import Generator
from openai import OpenAI
from .base import BaseLLM


class VLLMClient(BaseLLM):
    def __init__(self, settings):
        self.client = OpenAI(
            base_url=settings.backends.vllm.base_url,
            api_key="dummy",
        )
        self.model = settings.llm.model
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message["content"]

    def stream(self, prompt: str) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices:
                yield chunk.choices[0].delta.get("content", "")
