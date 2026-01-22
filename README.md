# 🧱 RAG KitCore

A modular Retrieval‑Augmented Generation (RAG) application built with FastAPI, a pluggable vectorstore, and a clean, extensible architecture.

This project was generated using the RAG KitCore Cookiecutter template, providing a production‑ready foundation for building scalable LLM‑powered systems.

---

## 🚀 Features

### 🔧 Architecture

- Clean, domain‑driven layout

- Config‑driven behavior (config.yaml)
- Structured logging
- Modular components:
  - `rag/` — RAG pipeline

  - `vectorstore/` — vector DB abstraction
  - `webapp/` — FastAPI backend
  - `scripts/` — ingestion utilities
  - `utils/` — helpers
  - `tests/` — pytest suite

🧠 RAG Pipeline

- Document ingestion

- Chunking

- Embedding

- Vectorstore indexing

- Retrieval

- Prompt construction

- LLM wrapper

🗄️ Vectorstore

Depending on your project configuration:

- FAISS‑like in‑memory store

- Qdrant (Docker)

- Weaviate (Docker)

🧩 LLM Providers

- OpenAI

- Azure OpenAI

- Ollama

- Dummy (offline testing)

🐳 Docker Support

- Optional Dockerized stack

- Auto‑configured vector DB containers

- FastAPI container

📦 Installation

1. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
⚙️ Configuration

All runtime settings are controlled via:
```bash
config.yaml
```
You can configure:

- LLM provider

- Embedding model

- Vectorstore backend

- Chunking parameters

- API keys

- Paths

Environment variables override config values when present.

---

## 📥 Ingest Documents

Place your documents inside:
```bash
data/
```
Then run:
```bash
python scripts/ingest.py
```
This will:

- load documents

- chunk them

- embed them

- store them in the vectorstore

---


## 🚀 Run the API

### Development mode
```bash
uvicorn webapp.api:app --reload
```

Open:
```bash
http://localhost:8000
```
Production mode (example)
```bash
uvicorn webapp.api:app --host 0.0.0.0 --port 8000
```

---


## 🐳 Run with Docker (if enabled)
```bash
docker compose up --build
```
This will start:

- FastAPI backend

- Vectorstore container (Qdrant/Weaviate if selected)

---


## 🧪 Testing

Run the test suite:
```bash
pytest -q
```

---

## 📁 Project Structure

```
rag_kitcore/
├── config/
├── data/
├── logging/
├── rag/
├── scripts/
│   └── ingest.py
├── tests/
├── utils/
├── vectorstore/
├── webapp/
│   ├── api.py
│   ├── routers/
│   ├── templates/   (if UI enabled)
│   └── static/      (if UI enabled)
│
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧩 Extending the System

You can easily extend:

### Add a new vectorstore

Create a new module under `vectorstore/` implementing the base interface.

### Add a new LLM provider

Add a wrapper under `rag/llm_providers/`.

### Add new ingestion logic

Modify or extend `scripts/ingest.py`.

### Add new API routes

Create routers under `webapp/routers/`.

---

## 🤝 Contributing

This project follows the structure and conventions of the RAG KitCore template.Feel free to extend it based on your needs.

---