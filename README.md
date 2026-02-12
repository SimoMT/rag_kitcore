# rag_kitcore

A modular, extensible Retrieval‑Augmented Generation (RAG) toolkit designed for clarity, maintainability, and production‑grade workflows.  
The project follows a clean architecture with separate subsystems for:

- **Indexing** (document ingestion, conversion, cleaning, chunking, embedding, vector store population)
- **Retrieval** (hybrid search, reranking, query pipelines)
- **LLM orchestration** (prompting, backends, inference)
- **Configuration** (typed settings, YAML + .env support)

This repository uses a `src/` layout and Poetry for dependency management.

---

## 📦 Installation

```bash
poetry install
poetry shell
```

---
## 📚 Subsystem Documentation
### Indexing Pipeline
Documentation for the indexing subsystem is available at:

```Code
src/rag_kitcore/rag/indexing/README.md
```
This includes:

* architecture overview
* module responsibilities
* configuration structure
* how to run the indexing pipeline
* how to extend converters, chunkers, embedders, and vector stores

---

## 🚀 Running the Indexing Pipeline
```bash
python -m rag_kitcore.rag.indexing
```
or via the CLI:

```bash
rag-index
```
---

## 🗂 Project Structure
```Code
rag_kitcore/
│
├── pyproject.toml
├── README.md
├── config/
│   ├── config.yaml
│   └── prompts.yaml
└── src/
    └── rag_kitcore/
        ├── logsys/
        ├── rag/
        │   └── conversion/
        │   └── embedding/
        │   └── indexing/
        │   └── pipelines/
        │   └── prompts/
        │   └── retrievers/
        └── core/
            └── bm_25/
            └── device/
            └── embeddings/
            └── llm/
            └── rerankers/
            └── vectorstore/
            └── exceptions.py
            └── settings.py
```

---

## 🧭 Status
Indexing subsystem: **complete and stable**
Retrieval subsystem: **in progress**  
LLM orchestration: **in progress**

---

## 🔢 ML Stack Versions (Pinned)

The backend uses a fully pinned ML stack to ensure reproducible builds and stable behavior across CPU and GPU environments.

| Component               | Version     | Notes |
|-------------------------|-------------|-------|
| PyTorch (CPU)           | 2.2.0       | Official CPU wheels |
| Transformers            | 4.37.2      | Compatible with ST 2.2.2 |
| Tokenizers              | 0.15.1      | Matches Transformers 4.37 |
| SentenceTransformers    | 2.2.2       | Stable, CrossEncoder supported |
| HuggingFace Hub         | 0.19.4      | Required for ST 2.2.2 (`cached_download`) |
| SentencePiece           | 0.1.99      | Required for HF models |
| scikit-learn            | 1.3.2       | Required for ST |
| nltk                    | 3.8.1       | Required for ST |
| Qdrant Client           | 1.7.3       | Matches Qdrant 1.7.x |
| FastAPI                 | 0.109.0     | Backend API |
| Uvicorn                 | 0.27.0      | ASGI server |

This stack is intentionally conservative to ensure stability during development.  
A future upgrade to SentenceTransformers 3.x will be handled separately.
