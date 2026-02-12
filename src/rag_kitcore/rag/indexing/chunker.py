from typing import List
from langchain_text_splitters  import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from rag_kitcore.core.exceptions import ChunkingError
from rag_kitcore.logsys.logger import get_logger
from rag_kitcore.rag.indexing.utils import (
    merge_small_chunks_documents,
    merge_adjacent_table_chunks_documents,
    is_index_chunk,
    enforce_max_tokens,
)

logger = get_logger(__name__)


def chunk_markdown(
    text: str,
    tokenizer,
    chunk_size: int,
    chunk_overlap: int,
    max_tokens: int = 512,
) -> List[Document]:
    """
    Convert cleaned markdown into structured, token-aware chunks.
    """
    try:
        if not text or not text.strip():
            logger.warning("chunk_markdown: received empty text")
            return []

        headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        md_docs = MarkdownHeaderTextSplitter(headers).split_text(text)
        logger.debug("Header split produced %d sections", len(md_docs))

        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        raw_chunks = splitter.split_documents(md_docs)
        logger.debug("Token split produced %d raw chunks", len(raw_chunks))

        processed = []
        for d in raw_chunks:
            if is_index_chunk(d.page_content):
                logger.debug("Skipping index-like chunk")
                continue

            header_values = [
                value for key, value in d.metadata.items()
                if key.startswith("H")
            ]
            if header_values:
                d.page_content = f"CONTEXT: {' > '.join(header_values)}\n{d.page_content}"

            processed.append(d)

        merged = merge_small_chunks_documents(processed)
        merged = merge_adjacent_table_chunks_documents(merged)

        final_docs = enforce_max_tokens(merged, tokenizer, max_tokens=max_tokens)

        for d in final_docs[:5]:
            tok_count = len(tokenizer.encode(d.page_content))
            logger.debug("Sample chunk token length: %d", tok_count)

        logger.info("Final chunk count: %d", len(final_docs))
        return final_docs

    except Exception as exc:
        logger.exception("Chunking failed")
        raise ChunkingError("Failed during markdown chunking") from exc
