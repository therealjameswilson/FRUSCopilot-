import streamlit as st
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.volume_suggester import suggest_documents

st.title("FRUS Compiler Copilot")

topic = st.text_input("Enter proposed FRUS volume topic")

if topic:
    results = suggest_documents(topic)

    st.subheader("Suggested Documents")

    for r in results:
        st.write("Doc ID:", r[0], "| relevance:", round(r[1], 3))
