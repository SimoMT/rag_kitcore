import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import time
import streamlit as st

from core.settings import Settings

settings = Settings.from_yaml()
from core.resources import load_resources
from rag.pipelines.extractor_rag import build_rag_chain
from webapp.ui_helpers import (
    render_chat_history,
    append_message,
)



# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="RAG Assistant",
    layout="wide",
    page_icon="✈️"
)

# -------------------------------------------------
# Load backend resources + RAG chain
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_backend():
    resources = load_resources()
    llm = resources["llm"]
    chain = build_rag_chain(llm, resources)
    return resources, chain


with st.spinner("Caricamento modelli e indici..."):
    resources, chain = init_backend()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("RAG AI Assistant")
st.caption(
    f"Modello LLM: **{settings.llm_model}** | Reranker: **{settings.reranker_model}**"
)

# -------------------------------------------------
# Chat history
# -------------------------------------------------
render_chat_history()

# -------------------------------------------------
# User input
# -------------------------------------------------
prompt = st.chat_input("Fai una domanda sui documenti tecnici...")

if prompt:
    append_message("user", prompt)

    # Render user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response container
    with st.chat_message("assistant"):
        start = time.time()

        placeholder = st.empty()
        full_response = ""

        try:
            # ---------------------------
            # Stream RAG pipeline output
            # ---------------------------
            for chunk in chain.stream(prompt):
                full_response += chunk
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            elapsed = time.time() - start
            st.markdown(
                f"<small style='color:grey'>⏱️ Generato in {elapsed:.2f}s</small>",
                unsafe_allow_html=True,
            )

            append_message("assistant", full_response)

        except Exception as e:
            st.error(f"Errore generazione: {e}")
