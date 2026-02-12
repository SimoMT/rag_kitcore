from abc import ABC, abstractmethod
from typing import Generator


class BaseLLM(ABC):

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return a full completion for the given prompt."""
        ...

    @abstractmethod
    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Yield chunks of text for streaming generation."""
        ...

    @abstractmethod
    def __call__(self, prompt: str, system_prompt: str | None = None):
        return self.generate(prompt, system_prompt)
