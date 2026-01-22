import re
from logsys import get_logger
from rag.indexing.processor import remove_short_lines

logger = get_logger(__name__)

def clean_markdown(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r" {3,}", " ", text)
    text = text.replace("||", "| |")
    text = re.sub(r"-{3,}", "-", text)
    text = re.sub(r"\.{3,}", ".", text)
    text = re.sub(r"<!--\s*image\s*-->", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()

    text = remove_short_lines(text)
    logger.debug("Markdown cleaned (%d chars)", len(text))
    return text
