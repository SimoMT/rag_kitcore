# 🔍 Local RAG Assistant

A fully local, privacy‑preserving Retrieval‑Augmented Generation (RAG) application built with Streamlit, Qdrant, BM25, a cross‑encoder reranker, and a lightweight LLM running through Ollama.

The system allows you to query your documents using hybrid search and receive structured, context‑aware answers.

---

## 🚀 Features

- **Hybrid Retrieval** 
Combines BM25 lexical search with vector search via Qdrant.

- **Reranking**  
Uses a MiniLM cross‑encoder to reorder retrieved documents by relevance.

- **Local LLM**  
Runs entirely on your machine using Ollama (no cloud calls, no API keys).

- **Embeddings**  
Uses sentence-transformers/all-MiniLM-L6-v2 for vector indexing.

- **Streamlit Web UI**  
Simple, interactive interface for querying your knowledge base.

- **Config‑driven**  
All settings stored in config.yaml.

- **Private by design**  
No external services required.

---

## 📦 Requirements

### Software
- Python 3.10+

- Pip or uv

- Ollama installed and running
https://ollama.com/download

- A local LLM model (e.g., llama3.2:1b)

- Qdrant (file‑based mode, no server required)

### Hardware
* CPU‑only is sufficient

* ~3 GB free disk space (Ollama + model)

---

## 🧠 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt>
```

### 3. Install Ollama
Download from:
https://ollama.com/download

### 4. Pull a lightweight model
```bash
ollama pull llama3.2:1b
```

5. Configure the application
Edit `config/config.yaml`:
```bash
llm_provider: ollama
llm_model: llama3.2:1b

embedding_model: sentence-transformers/all-MiniLM-L6-v2
reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

bm25_index: data/bm25_index
qdrant_url: http://localhost:6333
collection_name: rag_collection

```

---

## 🗂️ Preparing the Indexes

#### BM25
Ensure the BM25 index is stored in:

```bash
data/bm25_index
```
#### Qdrant
The application uses Qdrant in local file‑based mode, so no Docker or external service is required.

---

## ▶️ Running the Application

Make sure Ollama is running:
```bash
ollama serve
```

Then start the Streamlit app:
```bash
streamlit run webapp/streamlit_app.py
```

The UI will be available at:
```bash
http://localhost:8501
```

---

## 🧪 Quick Test
Try a simple query:

```bash
What is the description for ...?
```
If the system returns a precise value extracted from your documents, the RAG pipeline is working correctly.

---

## 🧱 Project Structure

```bash
.
├── config/
│   └── config.yaml                # Central configuration (LLM, embeddings, paths)
│
├── core/
│   ├── settings.py                # Loads config.yaml into a typed Settings object
│   └── resources.py               # Initializes LLM, embeddings, reranker, vector store
│
├── rag/
│   ├── retrievers/
│   │   ├── hybrid.py              # Hybrid retrieval logic (BM25 + Qdrant + reranker)
│   │   └── bm25.py                # BM25 index loading (if separated)
│   │
│   ├── pipelines/
│   │   └── rag_pipeline.py        # End‑to‑end RAG pipeline (retrieve → rerank → generate)
│   │
│   └── utils/
│       └── text_cleaning.py       # Optional preprocessing utilities
│
├── data/
│   ├── bm25_index/                # Serialized BM25 index
│   └── qdrant/                    # Local Qdrant storage (file‑based)
│
├── webapp/
│   ├── streamlit_app.py           # Main Streamlit UI
│   └── ui_helpers.py              # Formatting, layout, chat components
│
└── README.md

```

---
## 🔧 Troubleshooting

### ❌ “Connection refused” when generating answers

Ollama is not running.

Start it:
```bash
ollama serve
```

### ❌ “Model not found”

You need to pull the model:
```bash
ollama pull llama3.2:1b
```

### ❌ BM25 or Qdrant errors
Check that your indexes exist in the paths defined in config.yaml.
---

## 🧩 RAG Pipeline Diagram (ASCII)

The system uses a hybrid retrieval architecture combining BM25, vector search, and reranking before passing the final context to a local LLM. The diagram below summarizes the full flow:

```text
                         ┌──────────────────────────┐
                         │        User Query        │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │     Preprocessing        │
                         │ (normalization, cleanup) │
                         └─────────────┬────────────┘
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                         RETRIEVAL                            │
        └──────────────────────────────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┼──────────────────────────────┐
         ▼                             ▼                              ▼
┌──────────────────┐        ┌──────────────────┐          ┌──────────────────┐
│   BM25 Search    │        │ Vector Search    │          │   Metadata /     │
│ (lexical match)  │        │   (Qdrant)       │          │  Filters (opt.)  │
└─────────┬────────┘        └─────────┬────────┘          └─────────┬────────┘
          │                             │                             │
          └───────────────┬────────────┴───────────────┬─────────────┘
                          ▼                            ▼
                ┌──────────────────┐         ┌──────────────────┐
                │  BM25 Results    │         │ Vector Results    │
                └─────────┬────────┘         └─────────┬────────┘
                          │                            │
                          └───────────────┬────────────┘
                                          ▼
                         ┌──────────────────────────┐
                         │   Hybrid Combination     │
                         │ (weighted ensemble)      │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │        Reranker          │
                         │ (CrossEncoder MiniLM)    │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │   Top‑K Final Context    │
                         └─────────────┬────────────┘
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                         GENERATION                           │
        └──────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │      Local LLM (Ollama)  │
                         │   e.g., llama3.2:1b      │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │     Final Answer         │
                         └──────────────────────────┘
```


```text
                                ┌──────────────────────────────┐
                                │        Streamlit UI          │
                                │   webapp/streamlit_app.py    │
                                └───────────────┬──────────────┘
                                                │
                                                ▼
                                ┌──────────────────────────────┐
                                │       RAG Pipeline           │
                                │ rag/pipelines/rag_pipeline.py│
                                └───────────────┬──────────────┘
                                                │
                                                ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │                                RETRIEVAL                               │
        └────────────────────────────────────────────────────────────────────────┘
                                                │
                     ┌──────────────────────────┼──────────────────────────┐
                     ▼                          ▼                          ▼
        ┌──────────────────┐       ┌──────────────────────┐     ┌──────────────────────┐
        │ BM25 Retriever   │       │ Qdrant Vector Store  │     │  Preprocessing Utils │
        │ rag/retrievers/  │       │ core/resources.py    │     │ rag/utils/           │
        │ bm25.py          │       │                      │     │ text_cleaning.py     │
        └─────────┬────────┘       └────────────┬─────────┘     └────────────┬─────────┘
                  │                             │                           │
                  └───────────────┬─────────────┴──────────────┬────────────┘
                                  ▼                            ▼
                        ┌──────────────────┐         ┌──────────────────┐
                        │ BM25 Results     │         │ Vector Results    │
                        └─────────┬────────┘         └─────────┬────────┘
                                  │                            │
                                  └───────────────┬────────────┘
                                                  ▼
                                ┌──────────────────────────────┐
                                │ Hybrid Combiner (weights)    │
                                │ rag/retrievers/hybrid.py     │
                                └───────────────┬──────────────┘
                                                │
                                                ▼
                                ┌──────────────────────────────┐
                                │   Reranker (MiniLM CE)       │
                                │ core/resources.py            │
                                └───────────────┬──────────────┘
                                                │
                                                ▼
                                ┌──────────────────────────────┐
                                │     Final Context (Top‑K)    │
                                └───────────────┬──────────────┘
                                                │
                                                ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │                               GENERATION                               │
        └────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
                                ┌──────────────────────────────┐
                                │   Local LLM via Ollama       │
                                │ core/resources.py            │
                                └──────────────────────────────┘
```

## 🔄 Sequence Diagram — Request Flow
```text
User
 │
 │ 1. Enters query
 ▼
Streamlit UI (streamlit_app.py)
 │
 │ 2. Sends query to RAG pipeline
 ▼
RAG Pipeline (rag_pipeline.py)
 │
 │ 3. Preprocess query
 ▼
Hybrid Retriever (hybrid.py)
 │
 │ 4. BM25 search
 ▼
BM25 Index (data/bm25_index)
 │
 │ 5. Return lexical matches
 ▼
Hybrid Retriever
 │
 │ 6. Vector search
 ▼
Qdrant (data/qdrant/)
 │
 │ 7. Return vector matches
 ▼
Hybrid Retriever
 │
 │ 8. Combine + weight results
 ▼
Reranker (MiniLM CrossEncoder)
 │
 │ 9. Score and reorder documents
 ▼
RAG Pipeline
 │
 │ 10. Select Top‑K context
 ▼
Local LLM (Ollama)
 │
 │ 11. Generate final answer
 ▼
Streamlit UI
 │
 │ 12. Display answer to user
 ▼
User

```