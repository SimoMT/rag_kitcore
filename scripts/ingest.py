from vectorstore.loaders import load_text_file
from vectorstore.chunker import chunk_text
from vectorstore.embeddings import EmbeddingModel
from vectorstore.vector_db import get_vector_db
from vectorstore.index_builder import build_index
from config import load_config
from utils.helpers import ensure_dir


def main():
    config = load_config()
    embedder = EmbeddingModel(config.embeddings.model_name)
    vectordb = get_vector_db(config)

    ensure_dir("data")

    raw_text = load_text_file("data/sample.txt")
    chunks = chunk_text(raw_text)

    docs = [{"text": c, "id": i} for i, c in enumerate(chunks)]
    build_index(docs, embedder, vectordb)

    print("Index built successfully")


if __name__ == "__main__":
    main()
