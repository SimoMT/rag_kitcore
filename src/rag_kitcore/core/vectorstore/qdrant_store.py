from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_kitcore.logsys.logger import get_logger


logger = get_logger(__name__)

class QdrantStore:
    def __init__(self, url: str, collection_name: str):
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)

    def create(self, dim: int):
        # Drop if exists
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
        self.store = Qdrant.from_documents(
            documents=docs,
            embedding=embedder.model,
            url=self.url,
            collection_name=self.collection_name,
        )

    def as_retriever(self):
        return self.store.as_retriever()
