from __future__ import annotations

import re
from collections import defaultdict

from openai import OpenAI

from agents.retriever import search
from config import OPENAI_API_KEY


THEME_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "arms_control_nonproliferation": {
        "triggers": [
            "arms control",
            "nonproliferation",
            "nunn-lugar",
            "ctr",
            "strategic arms",
            "nuclear",
            "soviet weapons",
            "disarmament",
            "missile",
            "salt",
            "start",
            "abm",
            "inf treaty",
        ],
        "expansions": [
            "SALT negotiations strategic arms limitation talks ABM Treaty",
            "nuclear nonproliferation policy NPT safeguards IAEA",
            "strategic weapons reductions START INF verification compliance",
            "U.S.-Soviet arms control confidence-building measures",
        ],
    }
}


def suggest_documents(
    topic: str,
    top_k: int = 20,
    volume_slug: str | None = None,
    *,
    selected_volume: str | None = None,
) -> list[dict]:
    """Retrieve documents for a topic.

    `selected_volume` is accepted for compatibility with callers that pass the
    newer thematic-retrieval signature. It is currently unused in basic
    retrieval mode.
    """
    filters = {"volume_slug": volume_slug} if volume_slug else None
    return search(query=topic, top_k=top_k, filters=filters)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _detect_themes(topic: str, selected_volume: str | None = None) -> list[str]:
    haystack = _normalize_text(" ".join(filter(None, [topic, selected_volume or ""])))
    themes: list[str] = []
    for name, data in THEME_KEYWORDS.items():
        if any(trigger in haystack for trigger in data["triggers"]):
            themes.append(name)
    return themes


def build_thematic_queries(topic: str, selected_volume: str | None = None) -> list[tuple[str, str]]:
    themed_queries: list[tuple[str, str]] = [("primary", topic)]
    for theme in _detect_themes(topic, selected_volume=selected_volume):
        for expansion in THEME_KEYWORDS[theme]["expansions"]:
            themed_queries.append((theme, f"{topic} {expansion}".strip()))
    return themed_queries


def retrieve_thematic_documents(
    topic: str,
    *,
    selected_volume: str | None = None,
    top_k: int = 20,
    volume_slug: str | None = None,
) -> list[dict]:
    filters = {"volume_slug": volume_slug} if volume_slug else None
    per_doc: dict[str, dict] = {}
    theme_hits: dict[str, set[str]] = defaultdict(set)

    for theme, query in build_thematic_queries(topic, selected_volume=selected_volume):
        for item in search(query=query, top_k=top_k, filters=filters):
            key = item.get("chunk_id") or "|".join(
                [
                    str(item.get("volume_slug", "")),
                    str(item.get("document_number", "")),
                    str(item.get("history_state_url", "")),
                    str(item.get("title", "")),
                ]
            )

            existing = per_doc.get(key)
            if existing is None or item.get("score", 0.0) > existing.get("score", 0.0):
                per_doc[key] = {**item}

            theme_hits[key].add(theme)

    merged = []
    for key, item in per_doc.items():
        item["matched_themes"] = sorted(theme_hits[key])
        merged.append(item)

    merged.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    return merged[:top_k]


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=OPENAI_API_KEY)


def suggest_declassified_sources(topic: str, selected_volume: str | None = None, related_docs: list[dict] | None = None) -> str:
    related_docs = related_docs or []
    related_docs_lines = []
    for item in related_docs[:12]:
        related_docs_lines.append(
            "- "
            f"{item.get('title') or '(untitled)'} | "
            f"{item.get('volume_slug')} | "
            f"{item.get('history_state_url')}"
        )

    related_docs_block = "\n".join(related_docs_lines) if related_docs_lines else "- None provided"

    prompt = f"""
You are helping a FRUS historian researching: {topic}
Working volume context: {selected_volume or 'Not specified'}

Potentially related published FRUS documents:
{related_docs_block}

Provide concise, practical online declassified or born-unclassified sources.
Prioritize:
- National Archives (NARA catalog and digitized records)
- CIA FOIA Electronic Reading Room
- Department of State FOIA releases
- Presidential library digital archives
- DoD FOIA/release reading rooms
- Congressional hearings and committee records
- Government Publishing Office (GovInfo)

Return 8-12 bullet points. For each bullet include:
1) exact collection/document title,
2) direct URL to a specific declassified document (not just a homepage),
3) one sentence explaining why it would support FRUS selection decisions.

When possible, prefer links that could be printed or cited directly in FRUS (digitized scans, PDFs, or stable document pages).
"""
    response = _get_client().responses.create(model="gpt-5", input=prompt)
    return response.output_text


def suggest_classified_archives(topic: str) -> str:
    prompt = f"""
You are helping plan FRUS archival research for: {topic}

Suggest likely archival collections that may contain still-classified,
recently declassified, or hard-to-find records.
Prioritize likely U.S. government holdings and collection-level hints:
- Presidential library collections and NSC files
- State Department lot files
- RG 59 / NARA diplomatic records
- Defense Department archives
- CIA records and finding aids
- Joint Chiefs of Staff files

Return 8-12 bullet points with likely record groups/collection names.
"""
    response = _get_client().responses.create(model="gpt-5", input=prompt)
    return response.output_text
