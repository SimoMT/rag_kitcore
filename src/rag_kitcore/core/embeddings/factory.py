from rag_kitcore.core.embeddings.huggingface import HuggingFaceEmbedder


def create_embedder(settings, device: str = "cpu"):
    """
    For now we only support HuggingFace embeddings.
    Later you can add a backend switch if needed.
    """
    model_name = settings.models.embedding_model
    return HuggingFaceEmbedder(model_name, device=device)
