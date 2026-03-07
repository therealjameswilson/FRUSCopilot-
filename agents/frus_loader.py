from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from config import MAX_CHARS_PER_CHUNK

DOC_FILENAME_RE = re.compile(r"^d(\d+)\.md$", re.IGNORECASE)
VOLUME_START_YEAR_RE = re.compile(r"^frus(\d{4})", re.IGNORECASE)
DATE_LINE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"
)
YEAR_SPAN_RE = re.compile(r"frus(\d{4})(?:-(\d{2,4}))?", re.IGNORECASE)


@dataclass
class FrusDocument:
    volume_slug: str
    volume_title: str | None
    years: str | None
    administration: str | None
    document_number: str
    title: str | None
    date: str | None
    chapter_title: str | None
    section_title: str | None
    history_state_url: str
    source_path: str
    source_type: str
    text: str


def _normalize_text(raw: str) -> str:
    return raw.replace("\r\n", "\n").strip()


def infer_volume_slug(file_path: Path, volumes_root: Path) -> str:
    relative_path = file_path.relative_to(volumes_root)
    return relative_path.parts[0]


def infer_document_number(file_path: Path) -> str | None:
    match = DOC_FILENAME_RE.match(file_path.name)
    return match.group(1) if match else None


def infer_volume_start_year(volume_slug: str) -> int | None:
    match = VOLUME_START_YEAR_RE.match(volume_slug)
    return int(match.group(1)) if match else None


def is_supported_volume_slug(volume_slug: str, min_start_year: int = 1961) -> bool:
    start_year = infer_volume_start_year(volume_slug)
    return start_year is not None and start_year >= min_start_year


def build_history_state_url(volume_slug: str, document_number: str) -> str:
    return f"https://history.state.gov/historicaldocuments/{volume_slug}/d{document_number}"


def extract_title_and_date(text: str) -> tuple[str | None, str | None]:
    lines = [ln.strip().lstrip("#").strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0] if lines else None

    date = None
    for ln in lines[:30]:
        if DATE_LINE_RE.search(ln):
            date = ln
            break

    return title, date


def infer_years(volume_slug: str) -> str | None:
    match = YEAR_SPAN_RE.search(volume_slug)
    if not match:
        return None
    start = match.group(1)
    end = match.group(2)
    if not end:
        return start
    if len(end) == 2:
        century = start[:2]
        end = f"{century}{end}"
    return f"{start}-{end}"


def infer_administration(years: str | None) -> str | None:
    if not years:
        return None
    start = int(years.split("-")[0])
    if start >= 1993:
        return "Clinton"
    if start >= 1989:
        return "George H. W. Bush"
    if start >= 1981:
        return "Reagan"
    if start >= 1977:
        return "Carter"
    if start >= 1974:
        return "Ford"
    if start >= 1969:
        return "Nixon"
    if start >= 1963:
        return "Johnson"
    if start >= 1961:
        return "Kennedy"
    return None


def extract_section_metadata(text: str) -> tuple[str | None, str | None]:
    chapter = None
    section = None
    for ln in text.splitlines()[:80]:
        stripped = ln.strip()
        if stripped.startswith("## ") and not chapter:
            chapter = stripped[3:].strip()
        if stripped.startswith("### ") and not section:
            section = stripped[4:].strip()
        if chapter and section:
            break
    return chapter, section


def load_documents(volumes_root: Path, repo_root: Path) -> Iterator[FrusDocument]:
    for file_path in sorted(volumes_root.rglob("d*.md")):
        volume_slug = infer_volume_slug(file_path, volumes_root)
        if not is_supported_volume_slug(volume_slug):
            continue

        document_number = infer_document_number(file_path)
        if not document_number:
            continue

        text = _normalize_text(file_path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            continue

        title, date = extract_title_and_date(text)
        chapter_title, section_title = extract_section_metadata(text)
        source_path = str(file_path.relative_to(repo_root))
        years = infer_years(volume_slug)

        yield FrusDocument(
            volume_slug=volume_slug,
            volume_title=volume_slug,
            years=years,
            administration=infer_administration(years),
            document_number=document_number,
            title=title,
            date=date,
            chapter_title=chapter_title,
            section_title=section_title,
            history_state_url=build_history_state_url(volume_slug, document_number),
            source_path=source_path,
            source_type="frus_github",
            text=text,
        )


def _iter_text_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> Iterator[str]:
    if len(text) <= max_chars:
        yield text
        return

    paragraphs = text.split("\n\n")
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current and current_len + para_len > max_chars:
            yield "\n\n".join(current)
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        yield "\n\n".join(current)


def chunk_document(document: FrusDocument, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[dict]:
    pieces = list(_iter_text_chunks(document.text, max_chars=max_chars))
    chunks: list[dict] = []

    base_chunk_id = f"{document.volume_slug}:d{document.document_number}"
    for idx, piece in enumerate(pieces):
        chunk_id = base_chunk_id if len(pieces) == 1 else f"{base_chunk_id}#{idx + 1}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "volume_slug": document.volume_slug,
                "volume_title": document.volume_title,
                "years": document.years,
                "administration": document.administration,
                "document_number": document.document_number,
                "doc_number": document.document_number,
                "title": document.title,
                "doc_title": document.title,
                "date": document.date,
                "doc_date": document.date,
                "chapter_title": document.chapter_title,
                "section_title": document.section_title,
                "history_state_url": document.history_state_url,
                "source_url": document.history_state_url,
                "source_type": document.source_type,
                "source_path": document.source_path,
                "text": piece,
            }
        )

    return chunks


def write_chunks_jsonl(chunks: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
