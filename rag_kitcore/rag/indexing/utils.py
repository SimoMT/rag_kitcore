import re
from copy import copy
from typing import List
from langchain_core.documents import Document

from rag_kitcore.logsys.logger import get_logger


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


# ---------------------------------------------------------------------------
# Chunk merging
# ---------------------------------------------------------------------------

def merge_small_chunks_documents(
    docs: List[Document],
    min_length: int = DEFAULT_MIN_CHUNK_LENGTH,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[Document]:

    if not docs:
        return []

    merged: List[Document] = []

    for doc in docs:
        if not merged:
            merged.append(copy(doc))
            continue

        prev = merged[-1]
        prev_text = prev.page_content or ""
        curr_text = doc.page_content or ""

        prev_is_table = is_table_text(prev_text)
        curr_is_table = is_table_text(curr_text)
        combined_len = len(prev_text) + len(curr_text)

        if (
            not prev_is_table
            and not curr_is_table
            and len(prev_text) < min_length
            and combined_len <= max_chars
        ):
            logger.debug(
                "Merging small chunk: prev=%d, curr=%d, combined=%d",
                len(prev_text), len(curr_text), combined_len
            )
            merged[-1] = Document(
                page_content=f"{prev_text}\n{curr_text}",
                metadata={**prev.metadata, **doc.metadata},
            )
        else:
            merged.append(copy(doc))

    return merged


def merge_adjacent_table_chunks_documents(
    docs: List[Document],
    max_chars: int = DEFAULT_MAX_TABLE_CHARS,
) -> List[Document]:

    if not docs:
        return []

    merged: List[Document] = []

    for doc in docs:
        if not merged:
            merged.append(copy(doc))
            continue

        prev = merged[-1]
        prev_text = prev.page_content or ""
        curr_text = doc.page_content or ""

        if is_table_text(prev_text) and is_table_text(curr_text):
            combined_len = len(prev_text) + len(curr_text)
            if combined_len <= max_chars:
                logger.debug(
                    "Merging table chunks: prev=%d, curr=%d, combined=%d",
                    len(prev_text), len(curr_text), combined_len
                )
                merged[-1] = Document(
                    page_content=f"{prev_text}\n{curr_text}",
                    metadata={**prev.metadata, **doc.metadata},
                )
                continue

        merged.append(copy(doc))

    return merged


# ---------------------------------------------------------------------------
# Index detection
# ---------------------------------------------------------------------------

def is_index_chunk(
    text: str,
    dots_ratio_threshold: float = DEFAULT_INDEX_DOTS_RATIO,
    td_ratio_threshold: float = DEFAULT_INDEX_TD_RATIO,
) -> bool:

    if not text or not text.strip():
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    total = len(lines)

    if total == 0:
        return False

    # dotted index patterns
    dotted = sum(1 for line in lines if RE_INDEX_DOTS.match(line))
    if dotted / total > dots_ratio_threshold:
        return True

    # HTML table logic
    if RE_TABLE_TH.search(text):
        return False

    td_lines = [line for line in lines if RE_TABLE_TD.search(line)]
    if td_lines:
        td_ratio = len(td_lines) / total
        if td_ratio > td_ratio_threshold and any(
            RE_PAGE_NUM_IN_TD.search(line) for line in td_lines
        ):
            return True

    return False

# ---------------------------------------------------------------------------
# Enforxe Max token
# ---------------------------------------------------------------------------

def enforce_max_tokens(docs, tokenizer, max_tokens=512, safety_margin=50):
    """
    Ensure no chunk exceeds the embedding model's max token limit.
    If a chunk is too long, re-split it using the same tokenizer.
    """
    final_docs = []

    for d in docs:
        tokens = tokenizer.encode(d.page_content)

        if len(tokens) <= max_tokens:
            final_docs.append(d)
            continue

        # Re-split oversized chunk
        start = 0
        step = max_tokens - safety_margin  # leave room for headers, etc.

        while start < len(tokens):
            end = start + step
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)

            new_doc = Document(
                page_content=chunk_text,
                metadata=d.metadata.copy()
            )
            final_docs.append(new_doc)

            start += step

    return final_docs

# ---------------------------------------------------------------------------
# Cleaning utilities
# ---------------------------------------------------------------------------

def remove_short_lines(text: str, min_line_length: int = DEFAULT_MIN_LINE_LENGTH) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    filtered = [
        line for line in lines
        if "|" in line
        or RE_HTML_TAG.search(line)
        or len(line.strip()) >= min_line_length
    ]

    if len(filtered) < len(lines):
        logger.debug(
            "remove_short_lines: filtered %d/%d lines",
            len(lines) - len(filtered),
            len(lines),
        )

    return "\n".join(filtered)


def is_table_text(text: str) -> bool:
    return bool(text and "|" in text)