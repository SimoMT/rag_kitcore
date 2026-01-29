import os
import pickle
import streamlit as st
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from core.llm_factory import create_llm


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

def load_embeddings(settings, device: str):
    model_kwargs = {"trust_remote_code": True}

    if device.startswith("cuda"):
        model_kwargs["model_kwargs"] = {"dtype": torch.float16}

    return HuggingFaceEmbeddings(
        model_name=settings.models.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"device": device},
    )


# -------------------------------------------------
# Vector store
# -------------------------------------------------

def load_vector_store(settings, embeddings):
    client = QdrantClient(
        url=settings.paths.qdrant_url,
        prefer_grpc=False,
    )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.vectorstore.collection_name,
        embedding=embeddings,
    )


# -------------------------------------------------
# BM25
# -------------------------------------------------

def load_bm25(settings):
    path = settings.paths.bm25_index
    if not os.path.exists(path):
        raise FileNotFoundError(f"BM25 index not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------
# Reranker
# -------------------------------------------------

def load_reranker(settings, device: str):
    model_kwargs = {}

    if device.startswith("cuda"):
        model_kwargs["dtype"] = torch.float16

    return CrossEncoder(
        settings.models.reranker_model,
        device=device,
        model_kwargs=model_kwargs,
    )


# -------------------------------------------------
# LLM (via factory)
# -------------------------------------------------

def load_llm(settings):
    return create_llm(settings)


# -------------------------------------------------
# Orchestrator
# -------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_resources(settings):
    device = get_device()

    embeddings = load_embeddings(settings, device)
    vector_store = load_vector_store(settings, embeddings)
    bm25 = load_bm25(settings)
    reranker = load_reranker(settings, device)
    llm = load_llm(settings)

    return {
        "vector_store": vector_store,
        "bm25": bm25,
        "reranker": reranker,
        "llm": llm,
        "device": device,
    }
