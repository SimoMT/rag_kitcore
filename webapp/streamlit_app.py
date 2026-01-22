import streamlit as st
import requests

API_URL = "http://localhost:8000/query"  # or env-driven


def main():
    st.title("RAG KitCore - Streamlit UI")
    query = st.text_input("Ask something")
    if st.button("Send") and query:
        resp = requests.post(API_URL, json={"query": query})
        if resp.ok:
            st.write(resp.json().get("answer"))
        else:
            st.error("Error from backend")


if __name__ == "__main__":
    main()
