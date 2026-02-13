from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_kitcore.logsys.logger import get_logger

logger = get_logger(__name__)


class QdrantStore:
    def __init__(self, url: str, collection_name: str):
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self.store = None

    def create(self, dim: int):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def populate(self, docs, embedder):
        """
        Create a new Qdrant collection and populate it.
        """
        self.store = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embedder.model,
            url=self.url,
            collection_name=self.collection_name,
        )

    def load(self, embedder):
        """
        Load an existing Qdrant collection.
        """
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embedder.model,
        )

    def as_retriever(self):
        if self.store is None:
            raise RuntimeError(
                "QdrantStore.store is not initialized. "
                "Call populate() or load() before using as_retriever()."
            )
        return self.store.as_retriever()
