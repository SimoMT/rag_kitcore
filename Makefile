.PHONY: run test build up ingest format install clean

# Run the Streamlit webapp
run:
    streamlit run webapp/streamlit_app.py

# Run tests
test:
    pytest -q

# Build Docker images (if dockerized)
build:
    docker compose build

# Start services (if dockerized)
up:
    docker compose up -d

# Run ingestion pipeline
ingest:
    python scripts/ingest.py

# Format code
format:
    ruff check --fix .
    black .

# Install dependencies
install:
    pip install -r requirements.txt

# Clean caches and temporary files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
