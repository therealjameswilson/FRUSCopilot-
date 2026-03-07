from __future__ import annotations

from openai import OpenAI

from agents.retriever import search
from config import OPENAI_API_KEY


def suggest_documents(topic: str, top_k: int = 20, volume_slug: str | None = None) -> list[dict]:
    filters = {"volume_slug": volume_slug} if volume_slug else None
    return search(query=topic, top_k=top_k, filters=filters)


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=OPENAI_API_KEY)


def suggest_declassified_sources(topic: str) -> str:
    prompt = f"""
You are helping a FRUS historian researching: {topic}

Provide concise, practical online declassified or born-unclassified sources.
Prioritize:
- National Archives (NARA catalog and digitized records)
- CIA FOIA Electronic Reading Room
- Department of State FOIA releases
- Presidential library digital archives
- DoD FOIA/release reading rooms
- Congressional hearings and committee records
- Government Publishing Office (GovInfo)

Return 8-12 bullet points with specific collections/search entry points.
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
