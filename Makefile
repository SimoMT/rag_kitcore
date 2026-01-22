run:
    uvicorn webapp.api:app --reload

test:
    pytest -q

build:
    docker compose build

up:
    docker compose up

ingest:
    python scripts/ingest.py

format:
    ruff check --fix .
    black .

install:
    pip install -r requirements.txt
