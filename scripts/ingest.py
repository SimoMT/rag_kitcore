#!/usr/bin/env python3

import sys
from logsys import get_logger
from rag.indexing.pipeline import run_indexing
from core.exceptions import (
    DocumentConversionError,
    ChunkingError,
    VectorStoreError,
    RAGError,
)

logger = get_logger(__name__)


def main() -> int:
    logger.info("Starting ingestion pipeline")

    try:
        run_indexing()
        logger.info("Ingestion completed successfully")
        return 0

    except DocumentConversionError as exc:
        logger.error("Document conversion failed: %s", exc)
        return 1

    except ChunkingError as exc:
        logger.error("Chunking failed: %s", exc)
        return 2

    except VectorStoreError as exc:
        logger.error("Vectorstore operation failed: %s", exc)
        return 3

    except RAGError as exc:
        logger.error("RAG pipeline error: %s", exc)
        return 4

    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 99


if __name__ == "__main__":
    sys.exit(main())
