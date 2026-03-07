from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.retriever import search
from agents.volume_suggester import suggest_classified_archives, suggest_declassified_sources
from config import CHUNKS_PATH, EMBEDDINGS_DB_PATH, MANIFEST_PATH

st.set_page_config(page_title="FRUS Phase 1.1 Retriever", layout="wide")
st.title("FRUS Phase 1.1 Local Retrieval")

if not (CHUNKS_PATH.exists() and EMBEDDINGS_DB_PATH.exists() and MANIFEST_PATH.exists()):
    st.warning(
        "FRUS index not found. Run `python3 scripts/sync_frus_repo.py` then `python3 scripts/build_frus_index.py`."
    )
    st.stop()

query = st.text_input("Search topic")
volume_filter = st.text_input("Optional volume_slug filter (example: frus1969-76v34)")
top_k = st.slider("Top K", min_value=5, max_value=50, value=20, step=5)

if query:
    filters = {"volume_slug": volume_filter.strip()} if volume_filter.strip() else None
    results = search(query=query, top_k=top_k, filters=filters)

    st.subheader("FRUS Retrieval Results")
    if not results:
        st.info("No matching chunks found.")

    for item in results:
        st.markdown(f"**{item.get('title') or '(untitled)'}**")
        st.write(
            f"volume_slug: {item.get('volume_slug')} | "
            f"document_number: {item.get('document_number')} | "
            f"date: {item.get('date') or 'unknown'} | "
            f"score: {item.get('score', 0):.4f}"
        )
        st.markdown(f"[Public URL]({item.get('history_state_url')})")
        st.caption(item.get("source_path"))
        with st.expander("Text chunk"):
            st.write(item.get("text"))
        st.divider()

    with st.expander("Suggested declassified online sources"):
        st.write(suggest_declassified_sources(query))

    with st.expander("Likely classified archival collections"):
        st.write(suggest_classified_archives(query))
