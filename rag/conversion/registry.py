from .pdf_converter import PDFConverter
from .docx_converter import DocxConverter

CONVERTERS = [
    PDFConverter(),
    DocxConverter(),
]

def convert_file(path: str) -> str:
    for converter in CONVERTERS:
        if converter.can_handle(path):
            return converter.convert(path)
    raise ValueError(f"No converter available for file: {path}")
