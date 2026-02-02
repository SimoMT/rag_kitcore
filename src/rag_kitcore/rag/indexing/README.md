# 📘 README — Indexing Pipeline
## 📚 Overview
The indexing subsystem transforms raw documents into a searchable knowledge base using:

* **File conversion** (PDF, DOCX, TXT → Markdown)
* **Markdown cleaning**
* **Semantic chunking**
* **Embedding generation**
* **Vector store population (Qdrant)**
* **BM25 sparse index creation**

This pipeline is designed to be modular, backend‑agnostic, and easy to extend.

---

## 🧱 Architecture

```Code
input files
   ↓
file_resolver
   ↓
conversion registry (pdf/docx/txt)
   ↓
cleaner
   ↓
chunker
   ↓
embedder (HuggingFace)
   ↓
vector store (Qdrant)
   ↓
BM25 index
```

Each step is implemented as an independent module with a clear API.

---

## 🚀 Running the Indexing Pipeline

### Using the module runner
```Code
python -m rag_kitcore.rag.indexing
```

### Using the CLI command
```Code
rag-index
```

Both commands load configuration from:
```Code
config/config.yaml
config/prompts.yaml
```

---

## ⚙️ Configuration
The indexing pipeline is configured through the Settings model, which loads YAML files and optionally overrides values using .env.

### Example structure
```Yaml
rag:
  chunk_size: 250
  chunk_overlap: 50

paths:
  data_dir: "data/"
  bm25_index: "data/bm25_index"
  qdrant_url: "http://localhost:6333"

vectorstore:
  backend: "qdrant"
  collection_name: "rag_collection"

models:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```
---

## 🧩 Module Responsibilities
### file_resolver.py
* Scans the input directory
* Returns a list of supported file paths

### conversion/
* Registry-based converter system
* Supports PDF, DOCX, TXT
* Easy to extend with new formats

### cleaner.py
* Normalizes whitespace
* Removes noise
* Preserves tables and headings

### chunker.py
* Header-aware chunking
* Token-based splitting
* Table merging
* Index-like chunk filtering

### embeddings/
* HuggingFace embedder wrapper
* Exposes .embed_documents() and .embed_query()
* Provides .dimension

### vectorstore/
* Qdrant wrapper
* Creates collections
* Populates embeddings
* Exposes retriever interface

### bm25_builder.py
* Builds BM25 index
* Saves/loads index from disk

### pipeline.py
* Orchestrates the full indexing flow

---

## 🧪 Testing Individual Components
After installing the project:

```Code
poetry install
poetry shell
```

You can test modules directly:
```Python
from rag_kitcore.rag.indexing.chunker import chunk_markdown
from rag_kitcore.rag.indexing.cleaner import clean_markdown
from rag_kitcore.rag.conversion.registry import convert_file
```
---

## 🛠️ Extending the Indexing Pipeline
### Add a new file converter
Create a new class in conversion/ and register it in registry.py.

### Add a new vector store backend
Implement a class in core/vectorstore/ and update the factory.

### Add a new embedder
Implement BaseEmbedder and register it in the embedder factory.

---

## 📦 Outputs
The indexing pipeline produces:

* A populated **Qdrant collection**
* A serialized **BM25 index** at:

```Code
paths.bm25_index
```

These are used by the retrieval subsystem.

---

## 🧭 Troubleshooting
### Qdrant connection errors
Ensure Qdrant is running:

```Code
docker run -p 6333:6333 qdrant/qdrant
```

### Missing config
Ensure config/config.yaml exists and matches the Settings model.

### CLI path issues
Settings.from_yaml() resolves paths relative to the package root.