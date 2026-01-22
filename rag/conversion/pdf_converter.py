from rag.conversion.base import BaseConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from core.exceptions import DocumentConversionError
from logsys import get_logger

logger = get_logger(__name__)

class PDFConverter(BaseConverter):
    def can_handle(self, path: str) -> bool:
        return path.lower().endswith(".pdf")

    def convert(self, path: str) -> str:
        return _convert_pdf_to_markdown(path)

# --- Conversion -------------------------------------------------------------


def _convert_pdf_to_markdown(file_path: str) -> str:
    """
    Convert a PDF file to markdown using docling.
    Raises DocumentConversionError on failure.
    """
    logger.info("Starting PDF to Markdown conversion for %s", file_path)

    try:
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        logger.info("PDF to Markdown conversion completed. Raw length: %d chars", len(markdown_text))
        return markdown_text
    except Exception as exc:
        logger.exception("Error during PDF to Markdown conversion")
        raise DocumentConversionError(f"Failed to convert PDF '{file_path}' to markdown") from exc