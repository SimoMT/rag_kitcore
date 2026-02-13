from pathlib import Path
from typing import Dict

from rag_kitcore.rag.conversion.base import BaseConverter
from rag_kitcore.rag.conversion.pdf_converter import PDFConverter
from rag_kitcore.rag.conversion.docx_converter import DocxConverter
from rag_kitcore.rag.conversion.txt_converter import TXTConverter


CONVERTERS: Dict[str, BaseConverter] = {
    ".pdf": PDFConverter(),
    ".docx": DocxConverter(),
    ".txt": TXTConverter(),
}


def convert_file(path: str) -> str:
    ext = Path(path).suffix.lower()

    if ext not in CONVERTERS:
        raise ValueError(f"No converter registered for extension: {ext}")

    converter = CONVERTERS[ext]
    return converter(path)
