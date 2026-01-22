from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


SYSTEM_TEMPLATE = """
Sei un estrattore di dati tecnico di precisione.
Il tuo unico compito è trovare informazioni esatte all'interno di tabelle Markdown.

REGOLE DI ESTRAZIONE:
1. Identificazione Riga: Cerca nel contesto l'ID esatto richiesto (es. TS-XXX).
2. Lettura Orizzontale: Se trovi una riga che inizia con quell'ID, estrai il valore del campo richiesto.
3. Integrità del Dato: Riporta il testo esattamente come scritto.

CONTESTO:
{context}
"""

HUMAN_TEMPLATE = """
DOMANDA UTENTE:
{question}
"""


def build_prompt() -> ChatPromptTemplate:
    """Create the structured prompt for the extraction task."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", HUMAN_TEMPLATE),
    ])


def build_chain(llm):
    """
    Build the LCEL chain:
    prompt → llm → string parser
    """
    prompt = build_prompt()
    return prompt | llm | StrOutputParser()
