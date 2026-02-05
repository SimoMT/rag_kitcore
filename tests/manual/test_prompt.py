from rag_kitcore.core.settings import Settings
from rag_kitcore.rag.retrieval.pipeline import run_retrieval
from rag_kitcore.rag.prompting.pipeline import build_prompt

settings = Settings.from_yaml()

query = "What is a neural network?"
docs = run_retrieval(query, settings)

prompt = build_prompt(query, docs, settings)
print(prompt)
