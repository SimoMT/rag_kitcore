from rag_kitcore.core.settings import Settings

settings = Settings.from_yaml()

retriever = None
llm = None

def get_settings():
    return settings

def get_retriever():
    return retriever

def get_llm():
    return llm
