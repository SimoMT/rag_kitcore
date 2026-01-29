import streamlit as st

def render_chat_history():
    """Render all previous messages stored in session state."""
    messages = st.session_state.get("messages", [])
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def append_message(role, content):
    """Append a message to the chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": role, "content": content})


def render_sources(docs):
    """Render the list of retrieved documents with metadata."""
    with st.expander("📚 Fonti utilizzate per questa risposta"):
        for i, doc in enumerate(docs):
            score = doc.metadata.get("rerank_score", 0)
            st.markdown(f"**Fonte #{i+1}** (Score: {score:.4f})")
            st.caption(doc.page_content)
            st.divider()
