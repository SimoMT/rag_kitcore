import os
from logsys import get_logger
from core.settings import Settings

settings = Settings.from_yaml()

from rag.indexing.cleaner import clean_markdown
from rag.indexing.chunker import chunk_markdown
from rag.indexing.file_resolver import resolve_input_files
from rag.conversion.registry import convert_file
from rag.embedding.embedder import get_embedder
from vectorstore.qdrant_store import create_qdrant_collection, populate_qdrant
from retrievers.bm25 import build_bm25

logger = get_logger(__name__)

def run_indexing():
    input_files = resolve_input_files(settings.data_dir)
    if not input_files:
        logger.error("No files found in %s", settings.data_dir)
        return

    all_docs = []

    for file_path in input_files:
        raw = convert_file(file_path)
        clean = clean_markdown(raw)
        chunks = chunk_markdown(
            clean,
            model_name=settings.models.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        for d in chunks:
            d.metadata["source_file"] = os.path.basename(file_path)

        all_docs.extend(chunks)

    embedder = get_embedder(settings.models.embedding_model)
    dim = len(embedder.embed_query("test"))

    client = create_qdrant_collection(
        url=settings.paths.qdrant_url,
        collection=settings.vectorstore.collection_name,
        dim=dim,
    )

    populate_qdrant(client, settings.vectorstore.collection_name, embedder, all_docs)
    build_bm25(all_docs, settings.paths.bm25_index)

    logger.info("Indexing completed")
