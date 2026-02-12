from rag_kitcore.rag.conversion.base import BaseConverter


class TXTConverter(BaseConverter):
    def can_handle(self, path: str) -> bool:
        return path.lower().endswith(".txt")

    def convert(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def __call__(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
