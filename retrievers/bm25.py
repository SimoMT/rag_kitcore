import pickle
from langchain_community.retrievers import BM25Retriever

def build_bm25(docs, path: str):
    retriever = BM25Retriever.from_documents(docs)
    with open(path, "wb") as f:
        pickle.dump(retriever, f)
