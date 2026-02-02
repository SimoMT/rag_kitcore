import os
from transformers import AutoTokenizer

from rag_kitcore.logsys.logger import get_logger
from rag_kitcore.rag.indexing.cleaner import clean_markdown
from rag_kitcore.rag.indexing.chunker import chunk_markdown
from rag_kitcore.rag.indexing.file_resolver import resolve_input_files
from rag_kitcore.rag.indexing.bm25_builder import build_bm25_index, save_bm25_index
from rag_kitcore.rag.conversion.registry import convert_file
from rag_kitcore.core.embeddings.factory import create_embedder
from rag_kitcore.core.vectorstore.factory import create_vector_store

logger = get_logger(__name__)


def run_indexing(settings):
    # 1. Resolve input files
    input_files = resolve_input_files(settings.paths.data_dir)
    if not input_files:
        logger.error("No files found in %s", settings.paths.data_dir)
        return

    all_docs = []

    # 2. Tokenizer for chunking
    tokenizer = AutoTokenizer.from_pretrained(
        settings.models.embedding_model,
        trust_remote_code=True,
    )

    # 3. Convert → clean → chunk
    for file_path in input_files:
        raw = convert_file(file_path)
        clean = clean_markdown(raw)

        chunks = chunk_markdown(
            clean,
            tokenizer=tokenizer,
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
            max_tokens=getattr(settings.rag, "max_chunk_tokens", 512),
        )

        for d in chunks:
            d.metadata["source_file"] = os.path.basename(file_path)

        all_docs.extend(chunks)

    if not all_docs:
        logger.error("No chunks produced during indexing")
        return

    # 4. Embeddings
    embedder = create_embedder(settings, device="cpu")
    dim = embedder.dimension

    # 5. Vector store
    vector_store = create_vector_store(settings)
    vector_store.create(dim)
    vector_store.populate(all_docs, embedder)

    # 6. BM25
    bm25_retriever = build_bm25_index(all_docs)
    save_bm25_index(bm25_retriever, settings.paths.bm25_index)

    logger.info("Indexing completed")
