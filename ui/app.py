from __future__ import annotations

import json
import os
import inspect
import sys
from datetime import UTC, datetime

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import volume_suggester
from config import CHUNKS_PATH, FRUS_REPO_DIR, FRUS_VOLUMES_DIR, MANIFEST_PATH

# compatibility shim for missing get_retrieval_status
try:
    from agents.retriever import get_retrieval_status  # preferred
except ImportError:
    from pathlib import Path

    def _status_exists(pathlike) -> bool:
        try:
            return bool(pathlike) and Path(pathlike).exists()
        except Exception:
            return False

    def _count_jsonl_rows(pathlike) -> int:
        try:
            p = Path(pathlike)
            if not p.exists():
                return 0
            with p.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def get_retrieval_status():
        """
        Backward-compatible fallback for older/newer retriever modules.
        Returns a dict so existing status UI can keep working.
        """
        status = {
            "repo_exists": _status_exists(FRUS_REPO_DIR),
            "chunks_exists": _status_exists(CHUNKS_PATH),
            "chunks_count": _count_jsonl_rows(CHUNKS_PATH),
            "embeddings_exists": _status_exists(EMBEDDINGS_DB_PATH),
            "ready": _status_exists(CHUNKS_PATH) or _status_exists(EMBEDDINGS_DB_PATH),
            "frus_repo_dir": str(FRUS_REPO_DIR),
            "chunks_path": str(CHUNKS_PATH),
            "embeddings_db_path": str(EMBEDDINGS_DB_PATH),
        }

        if "FRUS_VOLUMES_DIR" in globals():
            status["volumes_dir_exists"] = _status_exists(FRUS_VOLUMES_DIR)
            status["frus_volumes_dir"] = str(FRUS_VOLUMES_DIR)

        if "MANIFEST_PATH" in globals():
            status["manifest_exists"] = _status_exists(MANIFEST_PATH)
            status["manifest_path"] = str(MANIFEST_PATH)

        return status


retrieve_thematic_documents = getattr(
    volume_suggester,
    "retrieve_thematic_documents",
    volume_suggester.suggest_documents,
)
retrieve_compiler_assist_documents = getattr(
    volume_suggester,
    "retrieve_compiler_assist_documents",
    None,
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


def build_suggestion_request_key(query: str, selected_volume: str | None, results: list[dict]) -> str:
    chunk_ids = [str(item.get("chunk_id") or item.get("history_state_url") or "") for item in results[:12]]
    return "|".join([query, selected_volume or "", *chunk_ids])


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

retrieval_status = get_retrieval_status()
status_code = retrieval_status.get("status")
chunk_count = int(retrieval_status.get("chunk_count") or 0)
chunk_path = retrieval_status.get("chunks_path")
emb_path = retrieval_status.get("embeddings_db_path")

if status_code == "missing_corpus":
    st.error(
        "FRUS corpus is missing; retrieval cannot run. "
        "Build it with `python3 scripts/build_frus_chunks.py --sync`. "
        f"Expected chunks path: {chunk_path}"
    )
elif status_code == "empty_corpus":
    st.error(
        "FRUS corpus loaded but contains zero chunks; retrieval cannot run. "
        "Rebuild with `python3 scripts/build_frus_chunks.py --sync`."
    )
elif status_code == "load_failed":
    st.error(
        "FRUS corpus failed to load. "
        f"Error: {retrieval_status.get('error')}"
    )
else:
    st.success(f"FRUS corpus ready: {chunk_count} chunks")

with st.expander("Retrieval/index status"):
    st.json(
        {
            "status": status_code,
            "chunks_path": chunk_path,
            "chunk_count": chunk_count,
            "distinct_volume_count": retrieval_status.get("distinct_volume_count"),
            "distinct_volumes_sample": retrieval_status.get("distinct_volumes_sample"),
            "embeddings_db_path": emb_path,
            "embeddings_db_exists": retrieval_status.get("embeddings_db_exists"),
            "embeddings_tables": retrieval_status.get("embeddings_tables"),
            "embeddings_row_counts": retrieval_status.get("embeddings_row_counts"),
            "chunk_embedding_alignment": retrieval_status.get("chunk_embedding_alignment"),
            "last_query_details": retrieval_status.get("last_query_details"),
        }
    )


query = st.text_input("Search topic")
mode = st.radio("Search mode", ["Exact Retrieval", "Compiler Assist (Inference Mode)"], index=1, horizontal=True)
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

    if mode == "Compiler Assist (Inference Mode)" and retrieve_compiler_assist_documents:
        call_kwargs["selected_volume"] = selected_volume
        call_kwargs["volume_slug"] = volume_slug
        assist_payload = retrieve_compiler_assist_documents(**call_kwargs)
        results = assist_payload.get("results", [])
        brief = assist_payload.get("brief", {})
        plan = assist_payload.get("plan", {})

        st.subheader("Compiler Assist Brief")
        st.write(f"**1. Exact phrase status:** {brief.get('exact_phrase_status', 'Unknown')}" )
        st.write(f"**2. Interpreted topic:** {brief.get('interpreted_topic', query)}")

        if assist_payload.get("used_inference"):
            st.warning(
                "Inference-expanded retrieval is active because exact retrieval was sparse. "
                "Only FRUS-indexed sources were searched."
            )
            with st.expander("Search planning details"):
                st.json(plan)

        st.markdown("**3. Top relevant documents**")
        for idx, doc in enumerate(brief.get("top_documents", []), start=1):
            st.markdown(
                f"{idx}. [{doc.get('title')}]({doc.get('url')}) "
                f"— {doc.get('volume_slug')} | {doc.get('date')}"
            )
            st.caption(doc.get("why_useful"))

        st.markdown("**4. Key themes and topics**")
        for theme in brief.get("key_themes", []):
            st.markdown(f"- {theme}")

        st.markdown("**5. Why these matter for compilation**")
        for line in brief.get("why_these_matter", []):
            st.markdown(f"- {line}")

        st.markdown("**6. Likely document families to pursue**")
        for family in brief.get("likely_document_families", []):
            st.markdown(f"- {family}")

        st.markdown("**7. Analog volumes or sections**")
        for analog in brief.get("analog_volumes_or_sections", []):
            st.markdown(f"- {analog}")

        st.subheader("Supporting FRUS Retrieval Results")
    else:
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
        runtime_status = get_retrieval_status()
        status = runtime_status.get("status")
        query_diag = runtime_status.get("last_query_details") or {}
        if status == "missing_corpus":
            st.error("No results because the FRUS corpus file is missing. Run `python3 scripts/build_frus_chunks.py --sync`.")
        elif status == "empty_corpus":
            st.error("No results because the FRUS corpus is empty. Rebuild with `python3 scripts/build_frus_chunks.py --sync`.")
        elif status == "load_failed":
            err = runtime_status.get("error")
            st.error(f"No results because corpus load failed: {err}")
        else:
            st.warning(
                "No matching chunks after retrieval. "
                f"Filtered candidates before semantic lookup: {query_diag.get('filtered_candidate_count', 0)}; "
                f"keyword hits: {query_diag.get('keyword_count', 0)}; "
                f"vector hits: {query_diag.get('vector_count', 0)}; "
                f"keyword fallback used: {query_diag.get('fallback_used', False)}."
            )

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
        st.markdown(f"[Public URL]({item.get('history_state_url') or item.get('source_url')})")
        st.caption(item.get("source_path") or item.get("source_file"))
        with st.expander("Text chunk"):
            st.write(item.get("text"))
        st.divider()

    request_key = build_suggestion_request_key(query, selected_volume, results)

    if st.session_state.get("declassified_request_key") != request_key:
        st.session_state["declassified_request_key"] = request_key
        st.session_state["declassified_response"] = None

    if st.session_state.get("classified_request_key") != request_key:
        st.session_state["classified_request_key"] = request_key
        st.session_state["classified_response"] = None

    with st.expander("Suggested declassified online sources (compiler-focused)"):
        if st.button("Generate declassified source suggestions", key="generate_declassified_sources"):
            with st.spinner("Generating declassified source suggestions..."):
                st.session_state["declassified_response"] = call_declassified_sources_suggester(
                    topic=query,
                    selected_volume=selected_volume,
                    related_docs=results,
                )

        declassified_response = st.session_state.get("declassified_response")
        if declassified_response:
            st.write(declassified_response)
        else:
            st.caption("Click the button to run declassified source search for this query.")

    with st.expander("Likely closed or partially closed archival repositories"):
        if st.button("Generate closed-archive suggestions", key="generate_classified_archives"):
            with st.spinner("Generating closed-archive suggestions..."):
                st.session_state["classified_response"] = call_classified_archives_suggester(
                    topic=query,
                    selected_volume=selected_volume,
                )

        classified_response = st.session_state.get("classified_response")
        if classified_response:
            st.write(classified_response)
        else:
            st.caption("Click the button to run closed-archive suggestion search for this query.")
