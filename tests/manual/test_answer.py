from rag_kitcore.core.settings import Settings
from rag_kitcore.rag.pipelines.answer import answer

settings = Settings.from_yaml("config/config.yaml")

query = "Explain what a neural network is"
response = answer(query, settings)

print("\nFinal answer:\n")
print(response)
