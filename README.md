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
        ├── rag/
        │   └── indexing/
        └── core/
```

---

## 🧭 Status
Indexing subsystem: **complete and stable**
Retrieval subsystem: **in progress**  
LLM orchestration: **in progress**

---

