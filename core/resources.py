import os
import pickle
import streamlit as st
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from core.settings import settings


# -------------------------------------------------
# Device selection
# -------------------------------------------------

def get_device(preferred_index: int = 0) -> str:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        index = preferred_index if preferred_index < count else 0
        return f"cuda:{index}"
    return "cpu"


# -------------------------------------------------
# Embeddings
# -------------------------------------------------

def load_embeddings(device: str):
    model_kwargs = {"trust_remote_code": True}

    if device.startswith("cuda"):
        model_kwargs["model_kwargs"] = {"dtype": torch.float16}

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"device": device},
    )


# -------------------------------------------------
# Vector store
# -------------------------------------------------

def load_vector_store(embeddings):
    client = QdrantClient(
        url=settings.qdrant_url,   # you use URL, not path
        prefer_grpc=False,
    )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )


# -------------------------------------------------
# BM25
# -------------------------------------------------

def load_bm25():
    path = settings.bm25_index
    if not os.path.exists(path):
        raise FileNotFoundError(f"BM25 index not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------
# Reranker
# -------------------------------------------------

def load_reranker(device: str):
    model_kwargs = {}

    if device.startswith("cuda"):
        model_kwargs["dtype"] = torch.float16

    return CrossEncoder(
        settings.reranker_model,
        device=device,
        model_kwargs=model_kwargs,
    )


# -------------------------------------------------
# LLM (Ollama)
# -------------------------------------------------

def load_llm():
    return ChatOllama(
        model=settings.llm_model,   # this is your field
        temperature=0.0,
        num_ctx=262144,
        keep_alive="5m",
        streaming=True,
    )


# -------------------------------------------------
# Orchestrator
# -------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_resources():
    device = get_device()

    embeddings = load_embeddings(device)
    vector_store = load_vector_store(embeddings)
    bm25 = load_bm25()
    reranker = load_reranker(device)
    llm = load_llm()

    return {
        "vector_store": vector_store,
        "bm25": bm25,
        "reranker": reranker,
        "llm": llm,
        "device": device,
    }
