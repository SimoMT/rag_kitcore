from typing import List
from sentence_transformers import CrossEncoder
from rag_kitcore.core.types import Document
from rag_kitcore.core.rerankers.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str, device: str):
        self.model = CrossEncoder(
            model_name,
            device=device,
            model_kwargs={"dtype": "float16"} if device.startswith("cuda") else {},
        )

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        texts = [d.page_content for d in docs]
        scores = self.model.predict([(query, t) for t in texts])
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]
