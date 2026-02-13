import os
import pickle

def load_bm25(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"BM25 index not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
