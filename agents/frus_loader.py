from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from config import MAX_CHARS_PER_CHUNK

DOC_FILENAME_RE = re.compile(r"^d(\d+)\.md$", re.IGNORECASE)
DATE_LINE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"
)


@dataclass
class FrusDocument:
    volume_slug: str
    document_number: str
    title: str | None
    date: str | None
    history_state_url: str
    source_path: str
    text: str


def _normalize_text(raw: str) -> str:
    return raw.replace("\r\n", "\n").strip()


def infer_volume_slug(file_path: Path, volumes_root: Path) -> str:
    relative_path = file_path.relative_to(volumes_root)
    return relative_path.parts[0]


def infer_document_number(file_path: Path) -> str | None:
    match = DOC_FILENAME_RE.match(file_path.name)
    return match.group(1) if match else None


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


def load_documents(volumes_root: Path, repo_root: Path) -> Iterator[FrusDocument]:
    for file_path in sorted(volumes_root.rglob("d*.md")):
        document_number = infer_document_number(file_path)
        if not document_number:
            continue

        text = _normalize_text(file_path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            continue

        volume_slug = infer_volume_slug(file_path, volumes_root)
        title, date = extract_title_and_date(text)
        source_path = str(file_path.relative_to(repo_root))

        yield FrusDocument(
            volume_slug=volume_slug,
            document_number=document_number,
            title=title,
            date=date,
            history_state_url=build_history_state_url(volume_slug, document_number),
            source_path=source_path,
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
                "document_number": document.document_number,
                "title": document.title,
                "date": document.date,
                "history_state_url": document.history_state_url,
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
