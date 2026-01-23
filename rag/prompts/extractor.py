# rag/prompts/extractor.py

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


def build_prompt():
    """
    Returns a formatter function that builds the final prompt string.
    This keeps the interface backend-agnostic and easy to extend.
    """

    def formatter(question: str, context: str) -> str:
        system_part = SYSTEM_TEMPLATE.format(context=context)
        human_part = HUMAN_TEMPLATE.format(question=question)
        return f"{system_part}\n{human_part}"

    return formatter
