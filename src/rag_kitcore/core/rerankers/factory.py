from rag_kitcore.core.settings import Settings
from rag_kitcore.core.rerankers.cross_encoder import CrossEncoderReranker


def create_reranker(settings: Settings):
    model_name = settings.models.reranker_model
    device = settings.backends.device
    return CrossEncoderReranker(model_name, device)
