from __future__ import annotations

import re
from collections import defaultdict
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.parse import quote_plus, urljoin
from urllib.request import urlopen

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

HISTORY_STATE_BASE_URL = "https://history.state.gov"
HISTORY_STATE_SEARCH_PATHS = [
    "/search?q={query}",
    "/search?query={query}",
]


class HistoryStateDocumentLinkParser(HTMLParser):
    """Extract document links from history.state.gov search pages."""

    def __init__(self) -> None:
        super().__init__()
        self.document_links: list[tuple[str, str]] = []
        self._capture_text = False
        self._current_href = ""
        self._current_text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        href = dict(attrs).get("href") or ""
        if "/historicaldocuments/" in href and re.search(r"/d\d+/?$", href):
            self._capture_text = True
            self._current_href = href
            self._current_text_chunks = []

    def handle_data(self, data: str) -> None:
        if self._capture_text:
            self._current_text_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._capture_text:
            return

        title = " ".join("".join(self._current_text_chunks).split())
        self.document_links.append((self._current_href, title))
        self._capture_text = False
        self._current_href = ""
        self._current_text_chunks = []


def _fetch_url(url: str) -> str:
    with urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def _score_history_state_hit(title: str, query: str) -> float:
    normalized_title = _normalize_text(title)
    query_terms = [term for term in re.split(r"\W+", _normalize_text(query)) if term]
    if not query_terms:
        return 0.0

    matched = sum(1 for term in query_terms if term in normalized_title)
    return matched / len(query_terms)


def search_history_state_documents(query: str, top_k: int = 20) -> list[dict]:
    """Search published FRUS documents on history.state.gov when local index has no hits."""
    documents: dict[str, dict] = {}

    for path_template in HISTORY_STATE_SEARCH_PATHS:
        search_url = urljoin(HISTORY_STATE_BASE_URL, path_template.format(query=quote_plus(query)))
        try:
            html = _fetch_url(search_url)
        except (TimeoutError, URLError, ValueError, OSError):
            continue

        parser = HistoryStateDocumentLinkParser()
        parser.feed(html)

        for raw_href, raw_title in parser.document_links:
            public_url = urljoin(HISTORY_STATE_BASE_URL, raw_href)
            match = re.search(r"/historicaldocuments/([^/]+)/d(\d+)/?$", public_url)
            if not match:
                continue

            volume_slug = match.group(1)
            document_number = match.group(2)
            title = raw_title or f"Document {document_number}"
            score = _score_history_state_hit(title=title, query=query)

            existing = documents.get(public_url)
            candidate = {
                "chunk_id": f"history-state|{volume_slug}|{document_number}",
                "title": title,
                "volume_slug": volume_slug,
                "document_number": document_number,
                "history_state_url": public_url,
                "source_path": "history.state.gov search",
                "text": "",
                "date": None,
                "score": score,
                "matched_themes": ["history_state_fallback"],
            }
            if existing is None or candidate["score"] > existing["score"]:
                documents[public_url] = candidate

    ranked = sorted(documents.values(), key=lambda row: row.get("score", 0.0), reverse=True)
    return ranked[:top_k]


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
    merged = merged[:top_k]

    if merged:
        return merged

    return search_history_state_documents(query=topic, top_k=top_k)


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
    response = _get_client().responses.create(model="gpt-5", input=prompt)
    return response.output_text


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
    response = _get_client().responses.create(model="gpt-5", input=prompt)
    return response.output_text
