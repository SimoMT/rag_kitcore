from rag_kitcore.core.embeddings.factory import create_embedder
from rag_kitcore.core.device.utils import get_device

def get_embedder(settings):
    device = get_device()
    return create_embedder(settings, device)
