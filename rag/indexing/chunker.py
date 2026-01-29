from typing import List
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer
from langchain_core.documents import Document

from core.exceptions import ChunkingError
from logsys import get_logger
from rag.indexing.processor import (
    merge_small_chunks_documents,
    merge_adjacent_table_chunks_documents,
    is_index_chunk,enforce_max_tokens,
)

logger = get_logger(__name__)


def chunk_markdown(
    text: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Convert cleaned markdown into structured, token-aware chunks.

    Steps:
    - Header-aware splitting
    - Token-aware splitting
    - Index-chunk filtering
    - Header context injection
    - Small-chunk merging
    - Table-chunk merging
    """
    try:
        if not text or not text.strip():
            logger.warning("chunk_markdown: received empty text")
            return []

        # 1. Header-aware split
        headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        md_docs = MarkdownHeaderTextSplitter(headers).split_text(text)
        logger.debug("Header split produced %d sections", len(md_docs))

        # 2. Token-aware split
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        raw_chunks = splitter.split_documents(md_docs)
        logger.debug("Token split produced %d raw chunks", len(raw_chunks))

        # 3. Process chunks
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

        logger.debug("Processed chunks: %d", len(processed))

        # 4. Merge small chunks
        merged = merge_small_chunks_documents(processed)

        # 5. Merge adjacent table chunks
        final_docs = merge_adjacent_table_chunks_documents(merged)

        # 6. Enforce max token limit AFTER all merging
        final_docs = enforce_max_tokens(final_docs, tokenizer, max_tokens=512)
        
        for d in final_docs[:5]:
            tok_count = len(tokenizer.encode(d.page_content))
            logger.debug(f"Sample chunk token length: {tok_count}")

        logger.info("Final chunk count: %d", len(final_docs))
        return final_docs

    except Exception as exc:
        logger.exception("Chunking failed")
        raise ChunkingError("Failed during markdown chunking") from exc
