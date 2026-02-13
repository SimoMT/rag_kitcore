from rag_kitcore.rag.indexing.pipeline import run_indexing
from rag_kitcore.core.settings import Settings

if __name__ == "__main__":
    print("Loading settings...")
    settings = Settings.from_yaml()
    print("Settings loaded:", settings)
    run_indexing(settings)
