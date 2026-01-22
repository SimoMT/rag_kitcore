from abc import ABC, abstractmethod

class BaseConverter(ABC):
    @abstractmethod
    def can_handle(self, path: str) -> bool:
        """Return True if this converter can process the file."""
        ...

    @abstractmethod
    def convert(self, path: str) -> str:
        """Return markdown or plain text extracted from the file."""
        ...
