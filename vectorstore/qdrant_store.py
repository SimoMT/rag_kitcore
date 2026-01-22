from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore

from core.exceptions import VectorStoreError
from logsys import get_logger

logger = get_logger(__name__)

def create_qdrant_collection(url: str, collection: str, dim: int):
    client = QdrantClient(url=url)

    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    return client


def populate_qdrant(client, collection: str, embedder, docs):
    try:
        store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embedder,
        )
        store.add_documents(docs)
        return store
    except Exception as exc:
        logger.exception("Failed to populate Qdrant")
        raise VectorStoreError from exc
