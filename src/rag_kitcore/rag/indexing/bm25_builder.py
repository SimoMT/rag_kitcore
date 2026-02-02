import pickle
from pathlib import Path
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document



def build_bm25_index(docs: List[Document]):
    """
    Build an in-memory BM25 retriever from a list of LangChain Documents.
    """
    return BM25Retriever.from_documents(docs)


def save_bm25_index(retriever: BM25Retriever, path: str | Path):
    """
    Persist a BM25 retriever to disk using pickle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(retriever, f)


def load_bm25_index(path: str | Path) -> BM25Retriever:
    """
    Load a BM25 retriever from disk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found at {path}")

    with open(path, "rb") as f:
        return pickle.load(f)
