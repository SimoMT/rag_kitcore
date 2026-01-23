# core/llm_ollama.py

from typing import Generator, Optional

import ollama


class OllamaClient:
    def __init__(self, settings) -> None:
        self.model: str = settings.llm.model
        self.temperature: float = settings.llm.temperature
        self.max_tokens: int = settings.llm.max_tokens
        # base_url is optional; if you need it:
        # ollama.Client(host=settings.backends.ollama.base_url)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        return response["message"]["content"]

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        for chunk in stream:
            content = chunk["message"].get("content", "")
            if content:
                yield content
