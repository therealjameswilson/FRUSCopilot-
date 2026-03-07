from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any

from openai import OpenAI

from agents.retriever import search
from config import OPENAI_API_KEY


SUGGESTER_TIMEOUT_SECONDS = float(os.getenv("FRUS_SUGGESTER_TIMEOUT_SECONDS", "25"))
INFERENCE_MIN_RESULTS = int(os.getenv("FRUS_INFERENCE_MIN_RESULTS", "5"))
INFERENCE_EXPANDED_TOP_K = int(os.getenv("FRUS_INFERENCE_EXPANDED_TOP_K", "40"))


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

PROCESS_KEYWORDS = {
    "nsc",
    "national security council",
    "white house",
    "interagency",
    "process",
    "management",
    "organization",
    "staffing",
    "decision",
    "coordination",
    "memorandum",
    "directive",
    "transition",
    "scowcroft",
    "briefing",
}


DEFAULT_DOCUMENT_TYPES = [
    "presidential memoranda",
    "NSC memoranda",
    "memoranda of conversation",
    "transition papers",
    "organizational guidance",
    "staffing memos",
]


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
    return OpenAI(api_key=OPENAI_API_KEY, timeout=SUGGESTER_TIMEOUT_SECONDS)


def _request_suggestions(prompt: str, fallback_label: str) -> str:
    try:
        response = _get_client().responses.create(model="gpt-5", input=prompt)
        return response.output_text
    except Exception as exc:
        return (
            f"{fallback_label} suggestion timed out or failed ({exc.__class__.__name__}). "
            "Please try again in a moment or reduce result scope."
        )


def _heuristic_search_plan(topic: str, selected_volume: str | None) -> dict[str, Any]:
    normalized = topic.strip()
    topic_lower = normalized.lower()

    rewrites = [normalized]
    if "scowcroft" in topic_lower:
        rewrites.extend(
            [
                "Brent Scowcroft",
                "National Security Council organization and management",
                "White House foreign policy process",
                "NSC staff Bush administration",
                "interagency coordination foreign policy",
            ]
        )
    elif "nsc" in topic_lower or "national security council" in topic_lower:
        rewrites.extend(
            [
                "National Security Council staff organization",
                "White House national security process",
                "NSC memoranda staffing guidance",
                "interagency foreign policy coordination",
            ]
        )
    else:
        rewrites.extend(
            [
                f"{normalized} organization and management of foreign policy",
                f"{normalized} White House process",
                f"{normalized} interagency coordination",
            ]
        )

    candidate_themes = [
        "organization and management of foreign policy",
        "interagency process",
        "White House decision-making",
        "national security process",
    ]
    entities = ["National Security Council", "White House", "Department of State"]
    if "scowcroft" in topic_lower:
        entities.insert(0, "Brent Scowcroft")
        entities.append("George H. W. Bush")

    return {
        "normalized_topic": normalized,
        "interpreted_meaning": (
            "Likely process/institutional query about how U.S. foreign policy was managed, "
            "including staffing, coordination, and decision flow."
        ),
        "candidate_entities": entities,
        "candidate_themes": candidate_themes,
        "query_rewrites": rewrites,
        "document_types_to_prioritize": DEFAULT_DOCUMENT_TYPES,
        "analog_volume_hints": [
            "published Organization and Management volumes",
            "White House and NSC process sections",
            "volumes with transition or institutional machinery discussions",
        ],
        "working_volume_context": selected_volume,
    }


def infer_search_plan(topic: str, selected_volume: str | None = None) -> dict[str, Any]:
    heuristic = _heuristic_search_plan(topic, selected_volume)
    if not OPENAI_API_KEY:
        return heuristic

    prompt = f"""
You are planning retrieval for a FRUS compiler-assistance system.
Return strict JSON with keys:
normalized_topic, interpreted_meaning, candidate_entities, candidate_themes,
query_rewrites, document_types_to_prioritize, analog_volume_hints.

Rules:
- This system can retrieve only from FRUS corpora (history.state.gov FRUS and HistoryAtState/frus chunks).
- Do not invent documents.
- Treat selected working volume as context only, especially if unpublished.
- query_rewrites should be retrieval-ready short queries.

Topic: {topic}
Selected working volume context: {selected_volume or 'Not provided'}
"""
    try:
        response = _get_client().responses.create(
            model="gpt-5",
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
        raw = response.output_text.strip()
        parsed = json.loads(raw)
        return {**heuristic, **parsed}
    except Exception:
        return heuristic


def _document_type_for_item(item: dict[str, Any]) -> str:
    text = _normalize_text(f"{item.get('title', '')} {item.get('text', '')[:300]}")
    if "memorandum of conversation" in text:
        return "memoranda of conversation"
    if "national security directive" in text or "directive" in text:
        return "directives"
    if "nsc" in text and "memorandum" in text:
        return "NSC memoranda"
    if "memorandum" in text:
        return "memoranda"
    if "telegram" in text:
        return "telegrams"
    if "minutes" in text:
        return "minutes"
    return "policy documents"


def merge_and_deduplicate_results(result_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rows in result_sets:
        for item in rows:
            key = item.get("chunk_id") or item.get("history_state_url") or item.get("title")
            existing = merged.get(str(key))
            if existing is None:
                merged[str(key)] = dict(item)
            else:
                existing["score"] = max(float(existing.get("score", 0.0)), float(item.get("score", 0.0)))
                themes = set(existing.get("matched_themes") or []) | set(item.get("matched_themes") or [])
                if themes:
                    existing["matched_themes"] = sorted(themes)
    return list(merged.values())


def rerank_for_compiler_use(results: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    entity_terms = {_normalize_text(e) for e in plan.get("candidate_entities", [])}
    theme_terms = {_normalize_text(t) for t in plan.get("candidate_themes", [])}

    reranked = []
    for item in results:
        haystack = _normalize_text(
            " ".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("section_title") or ""),
                    str(item.get("chapter_title") or ""),
                    str(item.get("text") or "")[:1000],
                ]
            )
        )
        bonus = 0.0
        if any(term and term in haystack for term in entity_terms):
            bonus += 0.18
        if any(term and term in haystack for term in theme_terms):
            bonus += 0.16
        if any(term in haystack for term in PROCESS_KEYWORDS):
            bonus += 0.14
        if item.get("source_type") in {"history_state", "frus_github"}:
            bonus += 0.05

        item = dict(item)
        item["document_type"] = _document_type_for_item(item)
        item["compiler_score"] = float(item.get("score", 0.0)) + bonus
        reranked.append(item)

    reranked.sort(key=lambda row: row.get("compiler_score", 0.0), reverse=True)
    return reranked


def _top_counts(values: list[str], top_n: int = 6) -> list[str]:
    return [name for name, _ in Counter([v for v in values if v]).most_common(top_n)]


def synthesize_compiler_brief(
    topic: str,
    selected_volume: str | None,
    plan: dict[str, Any],
    ranked_results: list[dict[str, Any]],
    exact_phrase_found: bool,
) -> dict[str, Any]:
    top = ranked_results[:10]
    if not top:
        return {
            "exact_phrase_status": f"Exact phrase '{topic}' not found in current retrieval scope.",
            "interpreted_topic": plan.get("interpreted_meaning") or plan.get("normalized_topic") or topic,
            "top_documents": [],
            "key_themes": plan.get("candidate_themes", []),
            "why_these_matter": [
                "No FRUS evidence was retrieved yet; refine topic terms or remove restrictive filters."
            ],
            "likely_document_families": plan.get("document_types_to_prioritize", DEFAULT_DOCUMENT_TYPES),
            "analog_volumes_or_sections": plan.get("analog_volume_hints", []),
            "selected_working_volume": selected_volume,
        }

    top_documents = [
        {
            "title": item.get("title") or "(untitled)",
            "volume_slug": item.get("volume_slug"),
            "date": item.get("date") or "unknown",
            "url": item.get("history_state_url"),
            "why_useful": (
                "Shows process/institutional context"
                if any(k in _normalize_text(item.get("text", "")[:1200]) for k in PROCESS_KEYWORDS)
                else "Provides related policy context and analog treatment in published FRUS volumes"
            ),
        }
        for item in top
    ]

    themes = _top_counts(
        [
            *[item.get("section_title") or "" for item in top],
            *[item.get("chapter_title") or "" for item in top],
            *[theme for item in top for theme in item.get("matched_themes", [])],
        ]
    )

    doc_families = _top_counts([item.get("document_type") or "" for item in top], top_n=8)

    return {
        "exact_phrase_status": (
            f"Exact phrase '{topic}' {'was' if exact_phrase_found else 'was not'} found in the top retrieved FRUS evidence."
        ),
        "interpreted_topic": plan.get("interpreted_meaning") or plan.get("normalized_topic") or topic,
        "top_documents": top_documents,
        "key_themes": themes or plan.get("candidate_themes", []),
        "why_these_matter": [
            "These published FRUS documents offer precedent for how similar themes were documented and edited.",
            "High-ranked items emphasize institutional process, staffing, and coordination signals relevant to compiler decisions.",
        ],
        "likely_document_families": doc_families or plan.get("document_types_to_prioritize", DEFAULT_DOCUMENT_TYPES),
        "analog_volumes_or_sections": plan.get("analog_volume_hints", []),
        "selected_working_volume": selected_volume,
    }


def retrieve_compiler_assist_documents(
    topic: str,
    *,
    selected_volume: str | None = None,
    top_k: int = 20,
    volume_slug: str | None = None,
    min_exact_results: int = INFERENCE_MIN_RESULTS,
) -> dict[str, Any]:
    filters = {"volume_slug": volume_slug} if volume_slug else None

    exact_results = search(query=topic, top_k=top_k, filters=filters, strategy="hybrid")
    exact_phrase_found = any(topic.lower() in _normalize_text(f"{r.get('title','')} {r.get('text','')}") for r in exact_results)

    used_inference = len(exact_results) < min_exact_results
    plan = infer_search_plan(topic, selected_volume=selected_volume) if used_inference else {
        "normalized_topic": topic,
        "interpreted_meaning": topic,
        "candidate_entities": [],
        "candidate_themes": [],
        "query_rewrites": [topic],
        "document_types_to_prioritize": DEFAULT_DOCUMENT_TYPES,
        "analog_volume_hints": [],
    }

    result_sets = [exact_results]
    if used_inference:
        for rewrite in plan.get("query_rewrites", [])[:8]:
            result_sets.append(search(query=rewrite, top_k=INFERENCE_EXPANDED_TOP_K, filters=filters, strategy="hybrid"))

    merged = merge_and_deduplicate_results(result_sets)
    ranked = rerank_for_compiler_use(merged, plan)[:top_k]
    brief = synthesize_compiler_brief(topic, selected_volume, plan, ranked, exact_phrase_found)

    return {
        "results": ranked,
        "brief": brief,
        "plan": plan,
        "used_inference": used_inference,
        "exact_phrase_found": exact_phrase_found,
        "exact_result_count": len(exact_results),
    }


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
You are serving a FRUS compiler who is actively compiling the current volume.
Research topic: {topic}
Current working volume from the dropdown: {selected_volume or 'Not specified'}

Potentially related published FRUS documents and prior-volume analogs:
{related_docs_block}

Provide concise, practical online declassified or born-unclassified sources that directly support compilation decisions for the current volume.
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
3) one sentence explaining why it would support FRUS selection decisions for the current volume,
4) when relevant, a short note connecting the source to precedent in earlier FRUS coverage.

When possible, prefer links that could be printed or cited directly in FRUS (digitized scans, PDFs, or stable document pages).
"""
    return _request_suggestions(prompt, fallback_label="Declassified source")


def suggest_classified_archives(topic: str, selected_volume: str | None = None) -> str:
    prompt = f"""
You are serving a FRUS compiler and planning archival research for: {topic}
Current working volume from the dropdown: {selected_volume or 'Not specified'}

Suggest likely closed, partially closed, or hard-to-access archival collections that may contain still-classified,
recently declassified, or hard-to-find records relevant to the active volume.
Prioritize likely U.S. government holdings and collection-level hints:
- Presidential library collections and NSC files
- State Department lot files
- RG 59 / NARA diplomatic records
- Defense Department archives
- CIA records and finding aids
- Joint Chiefs of Staff files

Return 8-12 bullet points with likely record groups/collection names.
For each bullet, add one short reason it is likely to matter for the active FRUS volume.
"""
    return _request_suggestions(prompt, fallback_label="Classified archive")
