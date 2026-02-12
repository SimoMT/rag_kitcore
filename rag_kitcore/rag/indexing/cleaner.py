import re
from rag_kitcore.logsys.logger import get_logger
from rag_kitcore.rag.indexing.utils import remove_short_lines

logger = get_logger(__name__)

# --- Pre-compiled Regex Patterns ---
RE_HTML_TAG = re.compile(r'<[^>]+>')
RE_INDEX_DOTS = re.compile(r'^.*\.{3,}\s*\d+\s*$')
RE_TABLE_TH = re.compile(r'<\s*th[^>]*>', re.IGNORECASE)
RE_TABLE_TD = re.compile(r'<\s*td[^>]*>', re.IGNORECASE)
RE_PAGE_NUM_IN_TD = re.compile(r'>\s*\d+\s*<')

DEFAULT_MIN_LINE_LENGTH = 3
DEFAULT_MIN_CHUNK_LENGTH = 100
DEFAULT_MAX_CHARS = 1500
DEFAULT_MAX_TABLE_CHARS = 1200
DEFAULT_INDEX_DOTS_RATIO = 0.6
DEFAULT_INDEX_TD_RATIO = 0.5

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
