from abc import ABC, abstractmethod

class BaseConverter(ABC):

    @abstractmethod
    def convert(self, path: str) -> str:
        """Return markdown or plain text extracted from the file."""
        ...

    @abstractmethod
    def __call__(self, path: str) -> str:
        """Convert a file at the given path into text."""
        ...
