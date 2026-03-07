from __future__ import annotations

import json
import os
import inspect
import sqlite3
import sys
from datetime import UTC, datetime

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import volume_suggester
from config import CHUNKS_PATH, EMBEDDINGS_DB_PATH, FRUS_REPO_DIR, FRUS_VOLUMES_DIR, MANIFEST_PATH


retrieve_thematic_documents = getattr(
    volume_suggester,
    "retrieve_thematic_documents",
    volume_suggester.suggest_documents,
)
suggest_classified_archives = volume_suggester.suggest_classified_archives
suggest_declassified_sources = volume_suggester.suggest_declassified_sources


def call_declassified_sources_suggester(
    *,
    topic: str,
    selected_volume: str | None,
    related_docs: list[dict],
) -> str:
    signature = inspect.signature(suggest_declassified_sources)
    call_kwargs: dict[str, object] = {}

    if "topic" in signature.parameters:
        call_kwargs["topic"] = topic
    elif "query" in signature.parameters:
        call_kwargs["query"] = topic
    else:
        first_param = next(iter(signature.parameters), None)
        if first_param:
            call_kwargs[first_param] = topic

    if "selected_volume" in signature.parameters:
        call_kwargs["selected_volume"] = selected_volume

    if "related_docs" in signature.parameters:
        call_kwargs["related_docs"] = related_docs
    elif "results" in signature.parameters:
        call_kwargs["results"] = related_docs

    return suggest_declassified_sources(**call_kwargs)


def call_classified_archives_suggester(*, topic: str, selected_volume: str | None) -> str:
    signature = inspect.signature(suggest_classified_archives)
    call_kwargs: dict[str, object] = {}

    if "topic" in signature.parameters:
        call_kwargs["topic"] = topic
    elif "query" in signature.parameters:
        call_kwargs["query"] = topic
    else:
        first_param = next(iter(signature.parameters), None)
        if first_param:
            call_kwargs[first_param] = topic

    if "selected_volume" in signature.parameters:
        call_kwargs["selected_volume"] = selected_volume

    return suggest_classified_archives(**call_kwargs)


TARGET_FRUS_VOLUMES: list[str] = [
    "Being Researched — 1917–1972, Volume II, Public Diplomacy, The Interwar Period",
    "Being Researched — 1917–1972, Volume III, Public Diplomacy, World War II",
    "Being Researched — 1956–1960, The Intelligence Community",
    "Being Researched — 1981–1988, Volume VIII, Western Europe, 1985–1988",
    "Being Researched — 1981–1988, Volume XXIII, Iran-Contra Affair, 1985–1988",
    "Being Researched — 1981–1988, Volume XXXVII, Trade; Monetary Policy; Industrialized Country Cooperation, 1985–1988",
    "Being Researched — 1981–1988, Volume XLII, Refugees and Immigration, 1975–1984",
    "Being Researched — 1981–1988, Volume XLV, Eastern Mediterranean",
    "Being Researched — 1989–1992, Volume I, Foundations of Foreign Policy; Public Diplomacy",
    "Being Researched — 1989–1992, Volume II, Organization and Management of Foreign Policy",
    "Being Researched — 1989–1992, Volume IV, Soviet Union, Russia, and Post-Soviet States: Policy",
    "Being Researched — 1989–1992, Volume V, Eastern Europe",
    "Being Researched — 1989–1992, Volume VI, Eastern Mediterranean",
    "Being Researched — 1989–1992, Volume VIII, Western Europe",
    "Being Researched — 1989–1992, Volume IX, Germany",
    "Being Researched — 1989–1992, Volume XIV, Arab-Israeli Dispute",
    "Being Researched — 1989–1992, Volume XV, South Asia",
    "Being Researched — 1989–1992, Volume XVI, Southeast Asia and the Pacific",
    "Being Researched — 1989–1992, Volume XVIII, Japan; Korea",
    "Being Researched — 1989–1992, Volume XX, North Africa; Sub-Saharan Africa",
    "Being Researched — 1989–1992, Volume XXII, Cuba; Haiti; Caribbean",
    "Being Researched — 1989–1992, Volume XXIII, Central America",
    "Being Researched — 1989–1992, Volume XXIV, Panama, 1981–1992",
    "Being Researched — 1989–1992, Volume XXV, South America",
    "Being Researched — 1989–1992, Volume XXVII, Arms Control and Nonproliferation",
    "Being Researched — 1989–1992, Volume XXX, Foreign Economic Policy",
    "Being Researched — 1989–1992, Volume XXXII, Iran",
    "Being Researched — 1993–2000, Volume I, Foundations of Foreign Policy",
    "Being Researched — 1993–2000, Volume IV, Foreign Economic Policy, 1993–1996",
    "Being Researched — 1993–2000, Volume XV, Wars in the Balkans, 1993–1995",
    "Being Researched — 1993–2000, Volume XX, Arms Control and Nonproliferation within the Former Soviet Union, December 1991–December 1994",
    "Being Researched — 1993–2000, Volume XXII, Europe: High-Level Contacts",
    "Being Researched — 1993–2000, Volume XXIV, Europe: Policy, 1997–2000",
    "Being Researched — 1993–2000, Volume XXV, Northern Ireland Peace Process",
    "Being Researched — 1993–2000, Volume XXVII, South Africa; Southern Africa",
    "Being Researched — 1993–2000, Volume XXVIII, Rwanda; Central Africa",
    "Being Researched — 1993–2000, Volume XXXII, Central America",
    "Planned — 1917–1972, Volume IV, Public Diplomacy, 1945–1952",
    "Planned — 1917–1972, Volume V, Public Diplomacy, 1953–1960",
    "Planned — 1989–1992, Volume XXVIII, Counternarcotics; Counterterrorism",
    "Planned — 1989–1992, Volume XXIX, Global Issues",
    "Planned — 1993–2000, Volume II, Organization and Management of Foreign Policy; Institutional Reform, 1992–1996",
    "Planned — 1993–2000, Volume III, Organization and Management of Foreign Policy; Institutional Reform, 1997–2000",
    "Planned — 1993–2000, Volume V, Foreign Economic Policy, 1997–2000",
    "Planned — 1993–2000, Volume VI, National Security Policy",
    "Planned — 1993–2000, Volume VII, Arms Control and Nonproliferation, 1993–1996",
    "Planned — 1993–2000, Volume VIII, Arms Control and Nonproliferation, 1997–2000",
    "Planned — 1993–2000, Volume IX, Counterterrorism Policy",
    "Planned — 1993–2000, Volume X, Global Issues: Transnational Security; United Nations; Multilateral Peacekeeping",
    "Planned — 1993–2000, Volume XI, Global Issues: Global Programs",
    "Planned — 1993–2000, Volume XII, Global Issues: Rights and Governance",
    "Planned — 1993–2000, Volume XIII, Global Issues: Transnational Commons",
    "Planned — 1993–2000, Volume XIV, Public Diplomacy",
    "Planned — 1993–2000, Volume XVI, Wars in the Balkans, 1995–2000",
    "Planned — 1993–2000, Volume XVII, North Atlantic Treaty Organization; European Security",
    "Planned — 1993–2000, Volume XVIII, Russia: High-Level Contacts",
    "Planned — 1993–2000, Volume XIX, Russia: Policy",
    "Planned — 1993–2000, Volume XXI, Ukraine; Belarus; Moldova; Transcaucasus; Central Asia",
    "Planned — 1993–2000, Volume XXIII, Europe: Policy, 1993–1996",
    "Planned — 1993–2000, Volume XXVI, Northern Africa",
    "Planned — 1993–2000, Volume XXIX, Africa Region; Western Africa; Eastern Africa",
    "Planned — 1993–2000, Volume XXX, North America",
    "Planned — 1993–2000, Volume XXXI, Cuba; Haiti; Caribbean",
    "Planned — 1993–2000, Volume XXXIII, South America; Latin America Region, 1993–1996",
    "Planned — 1993–2000, Volume XXXIV, South America; Latin America Region, 1997–2000",
    "Planned — 1993–2000, Volume XXXV, Middle East Peace Process, 1993–1996",
    "Planned — 1993–2000, Volume XXXVI, Middle East Peace Process, 1997–2000",
    "Planned — 1993–2000, Volume XXXVII, Iraq, 1993–1996",
    "Planned — 1993–2000, Volume XXXVIII, Iraq, 1997–2000",
    "Planned — 1993–2000, Volume XXXIX, Iran",
    "Planned — 1993–2000, Volume XL, Middle East Region; Arabian Peninsula",
    "Planned — 1993–2000, Volume XLI, China, 1993–1996",
    "Planned — 1993–2000, Volume XLII, China, 1997–2000",
    "Planned — 1993–2000, Volume XLIII, Japan; Korean Peninsula, 1993–1996",
    "Planned — 1993–2000, Volume XLIV, Japan; Korean Peninsula, 1997–2000",
    "Planned — 1993–2000, Volume XLV, South Asia, 1993–1996",
    "Planned — 1993–2000, Volume XLVI, South Asia, 1997–2000",
    "Planned — 1993–2000, Volume XLVII, Mainland Southeast Asia; East Asia Region",
    "Planned — 1993–2000, Volume XLVIII, Indonesia; Philippines; Oceania; Pacific Region",
]


def ensure_local_index_files() -> bool:
    created_any = False

    if not CHUNKS_PATH.exists():
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHUNKS_PATH.write_text("", encoding="utf-8")
        created_any = True

    if not EMBEDDINGS_DB_PATH.exists():
        EMBEDDINGS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(EMBEDDINGS_DB_PATH))
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS embeddings (chunk_id TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
            conn.commit()
        finally:
            conn.close()
        created_any = True

    if created_any:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1.1",
            "built_at": datetime.now(UTC).isoformat(),
            "repo_path": str(FRUS_REPO_DIR),
            "volumes_path": str(FRUS_VOLUMES_DIR),
            "chunk_count": 0,
            "volume_count": 0,
            "chunks_path": str(CHUNKS_PATH),
            "embeddings_path": str(EMBEDDINGS_DB_PATH),
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return created_any

st.set_page_config(page_title="FRUS Compiler Copilot Beta", layout="wide")
st.title("FRUS Compiler Copilot Beta")
st.markdown(
    """
### Mission: serve the FRUS compiler working on the current in-progress volume
- Use the volume dropdown to anchor all help to the exact volume currently being compiled.
- Surface relevant themes, topics, and useful analogs from earlier FRUS volumes.
- Highlight prior-volume context (for example, 1960s economics coverage when compiling 1990s economics).
- Return ranked chunks with source metadata and direct public URLs to support document selection decisions.
- Suggest declassified online records and likely closed repositories where additional records may exist.
"""
)

created_placeholder_index = ensure_local_index_files()

if created_placeholder_index:
    st.info(
        "Created missing index files so the app can start. "
        "Run `python3 scripts/sync_frus_repo.py` then `python3 scripts/build_frus_index.py` "
        "to build a real FRUS index."
    )

query = st.text_input("Search topic")
selected_volume = st.selectbox(
    "Current FRUS volume being compiled",
    options=TARGET_FRUS_VOLUMES,
    index=None,
    placeholder="Choose a volume",
)
volume_filter = st.text_input("Optional volume_slug filter (example: frus1969-76v34)")
top_k = st.slider("Top K", min_value=5, max_value=50, value=20, step=5)

if selected_volume:
    st.caption(f"Selected working volume: {selected_volume}")

if query:
    volume_slug = volume_filter.strip() or None
    call_kwargs: dict[str, object] = {
        "topic": query,
        "top_k": top_k,
    }
    retrieval_signature = inspect.signature(retrieve_thematic_documents)

    if "selected_volume" in retrieval_signature.parameters:
        call_kwargs["selected_volume"] = selected_volume

    if "filters" in retrieval_signature.parameters:
        call_kwargs["filters"] = {"volume_slug": volume_slug} if volume_slug else None
    elif "volume_slug" in retrieval_signature.parameters:
        call_kwargs["volume_slug"] = volume_slug

    results = retrieve_thematic_documents(**call_kwargs)

    st.subheader("FRUS Retrieval Results")
    if not results:
        st.info("No matching chunks found.")

    for item in results:
        themes = item.get("matched_themes") or ["primary"]
        st.markdown(f"**{item.get('title') or '(untitled)'}**")
        st.write(
            f"volume_slug: {item.get('volume_slug')} | "
            f"document_number: {item.get('document_number')} | "
            f"date: {item.get('date') or 'unknown'} | "
            f"score: {item.get('score', 0):.4f}"
        )
        st.caption(f"Matched themes/topics: {', '.join(themes)}")
        st.markdown(f"[Public URL]({item.get('history_state_url')})")
        st.caption(item.get("source_path"))
        with st.expander("Text chunk"):
            st.write(item.get("text"))
        st.divider()

    with st.expander("Suggested declassified online sources (compiler-focused)"):
        st.write(
            call_declassified_sources_suggester(
                topic=query,
                selected_volume=selected_volume,
                related_docs=results,
            )
        )

    with st.expander("Likely closed or partially closed archival repositories"):
        st.write(
            call_classified_archives_suggester(
                topic=query,
                selected_volume=selected_volume,
            )
        )
