from typing import List
from rag_kitcore.core.settings import Settings
from rag_kitcore.core.types import Document


def build_prompt(query: str, docs: List[Document], settings: Settings) -> str:
    """
    Build a citation-aware prompt.
    Each document is labeled [i] so the LLM can reference it.
    """

    # Load template from settings
    template = settings.prompts.extractor.system

    # Build context block
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        block = f"Document {i}:\n{doc.page_content}"
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    # Fill template
    prompt = template.format(
        context=context_str,
        question=query
    )

    return prompt
