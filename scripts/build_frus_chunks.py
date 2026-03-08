from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNKS_PATH, FRUS_GIT_URL, FRUS_REPO_DIR, FRUS_VOLUMES_DIR

YEAR_SPAN_RE = re.compile(r"(19\d{2}|20\d{2})(?:\D+(\d{2,4}))?")
DOC_ID_RE = re.compile(r"^d\d+[a-z]?$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
NOTE_MARKERS = ("Editorial Note", "Footnote", "Source:", "\n* ")


@dataclass
class BuildStats:
    volumes_kept: int = 0
    documents_chunked: int = 0
    chunks_written: int = 0
    files_skipped: int = 0


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def sync_frus_repo() -> None:
    FRUS_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not FRUS_REPO_DIR.exists():
        print(f"Cloning FRUS repository into {FRUS_REPO_DIR}...")
        run(["git", "clone", "--depth", "1", "--filter=blob:none", FRUS_GIT_URL, str(FRUS_REPO_DIR)])
    else:
        print(f"Updating FRUS repository in {FRUS_REPO_DIR}...")
        run(["git", "-C", str(FRUS_REPO_DIR), "fetch", "origin"])
        run(["git", "-C", str(FRUS_REPO_DIR), "pull", "--ff-only", "origin"])


def normalize_ws(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def find_first_text(parent: ET.Element, names: tuple[str, ...]) -> str:
    for el in parent.iter():
        if local_name(el.tag) in names:
            txt = normalize_ws(" ".join(el.itertext()))
            if txt:
                return txt
    return ""


def parse_years(volume_id: str, volume_title: str, source_name: str) -> tuple[int | None, int | None]:
    for candidate in (volume_id, volume_title, source_name):
        match = YEAR_SPAN_RE.search(candidate)
        if not match:
            continue
        start = int(match.group(1))
        end_raw = match.group(2)
        if not end_raw:
            return start, start
        if len(end_raw) == 2:
            end = int(f"{str(start)[:2]}{end_raw}")
        else:
            end = int(end_raw)
        return start, end
    return None, None


def extract_volume_metadata(root: ET.Element, file_path: Path) -> tuple[str, str, int | None, int | None, str | None]:
    volume_id = root.attrib.get("{http://www.w3.org/XML/1998/namespace}id") or file_path.stem
    title = find_first_text(root, ("title",)) or file_path.stem
    admin = ""
    lowered = title.lower()
    if "kennedy" in lowered:
        admin = "Kennedy"
    elif "johnson" in lowered:
        admin = "Johnson"
    elif "nixon" in lowered:
        admin = "Nixon"
    elif "ford" in lowered:
        admin = "Ford"
    elif "carter" in lowered:
        admin = "Carter"
    elif "reagan" in lowered:
        admin = "Reagan"
    elif "bush" in lowered:
        admin = "George H. W. Bush"
    elif "clinton" in lowered:
        admin = "Clinton"

    start, end = parse_years(volume_id, title, file_path.name)
    return volume_id, title, start, end, (admin or None)


def find_document_divs(root: ET.Element) -> list[ET.Element]:
    docs: list[ET.Element] = []
    for div in root.iter():
        if local_name(div.tag) != "div":
            continue
        div_type = (div.attrib.get("type") or "").lower()
        xml_id = div.attrib.get("{http://www.w3.org/XML/1998/namespace}id") or ""
        if div_type == "document" or DOC_ID_RE.match(xml_id):
            docs.append(div)
    return docs


def extract_doc_date(div: ET.Element) -> str | None:
    for el in div.iter():
        if local_name(el.tag) in {"date", "docDate", "dateline"}:
            txt = normalize_ws(" ".join(el.itertext()))
            if txt:
                return txt
    return None


def extract_document_record(div: ET.Element, root: ET.Element, volume_id: str, volume_title: str, start_year: int | None, end_year: int | None, administration: str | None, source_file: Path) -> dict | None:
    xml_id = div.attrib.get("{http://www.w3.org/XML/1998/namespace}id") or ""
    doc_id = xml_id if DOC_ID_RE.match(xml_id) else ""
    if not doc_id:
        n_attr = (div.attrib.get("n") or "").strip()
        if n_attr.isdigit():
            doc_id = f"d{n_attr}"
    if not doc_id:
        return None

    doc_title = find_first_text(div, ("head", "title")) or "Untitled document"
    section_title = find_first_text(div, ("subhead", "head")) or None
    chapter_title = find_first_text(div, ("head",)) or None
    compilation_title = volume_title

    text = normalize_ws(" ".join(div.itertext()))
    if not text:
        return None

    note_hits = sum(marker.lower() in text.lower() for marker in NOTE_MARKERS)
    if note_hits >= 3 and len(text.split()) < 220:
        return None

    return {
        "volume_id": volume_id,
        "volume_title": volume_title,
        "volume_start_year": start_year,
        "volume_end_year": end_year,
        "administration": administration,
        "document_id": doc_id,
        "document_title": doc_title,
        "chapter_title": chapter_title,
        "compilation_title": compilation_title,
        "section_title": section_title,
        "doc_date": extract_doc_date(div),
        "source_url": f"https://history.state.gov/historicaldocuments/{volume_id}/{doc_id}",
        "source_file": str(source_file),
        "text": text,
    }


def chunk_words(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, chunk_size_words - overlap_words)
    for start in range(0, len(words), step):
        part = words[start : start + chunk_size_words]
        if not part:
            break
        chunks.append(" ".join(part))
        if start + chunk_size_words >= len(words):
            break
    return chunks



def infer_start_year_from_path(file_path: Path) -> int | None:
    for part in file_path.parts[::-1]:
        match = re.search(r"frus(19\d{2}|20\d{2})", part, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def parse_volume_file(file_path: Path, chunk_size_words: int, overlap_words: int) -> tuple[list[dict], bool]:
    try:
        root = ET.parse(file_path).getroot()
    except Exception:
        return [], False

    volume_id, volume_title, start_year, end_year, administration = extract_volume_metadata(root, file_path)
    if start_year is None or start_year < 1961:
        return [], True

    output: list[dict] = []
    docs = find_document_divs(root)
    for div in docs:
        record = extract_document_record(
            div,
            root,
            volume_id=volume_id,
            volume_title=volume_title,
            start_year=start_year,
            end_year=end_year,
            administration=administration,
            source_file=file_path,
        )
        if not record:
            continue

        pieces = chunk_words(record["text"], chunk_size_words=chunk_size_words, overlap_words=overlap_words)
        for idx, piece in enumerate(pieces, start=1):
            chunk_id = f"{volume_id}:{record['document_id']}:{idx}"
            output.append(
                {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "text": piece,
                    "volume_id": record["volume_id"],
                    "volume_slug": record["volume_id"],
                    "volume_title": record["volume_title"],
                    "volume_start_year": record["volume_start_year"],
                    "volume_end_year": record["volume_end_year"],
                    "document_id": record["document_id"],
                    "document_number": record["document_id"].lstrip("d"),
                    "document_title": record["document_title"],
                    "title": record["document_title"],
                    "chapter_title": record["chapter_title"],
                    "compilation_title": record["compilation_title"],
                    "section_title": record["section_title"],
                    "source_url": record["source_url"],
                    "history_state_url": record["source_url"],
                    "source_file": record["source_file"],
                    "source_path": str(file_path.relative_to(FRUS_REPO_DIR)),
                    "chunk_index": idx,
                    "administration": record["administration"],
                    "doc_date": record["doc_date"],
                    "date": record["doc_date"],
                }
            )
    return output, True


def build_corpus(output_path: Path, chunk_size_words: int, overlap_words: int) -> BuildStats:
    stats = BuildStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    volume_ids: set[str] = set()

    with output_path.open("w", encoding="utf-8") as out:
        for xml_file in sorted(FRUS_VOLUMES_DIR.rglob("*.xml")):
            inferred_year = infer_start_year_from_path(xml_file)
            if inferred_year is not None and inferred_year < 1961:
                continue
            chunks, parsed_ok = parse_volume_file(xml_file, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
            if not parsed_ok:
                stats.files_skipped += 1
                print(f"SKIP (parse): {xml_file}")
                continue
            if not chunks:
                continue
            vol = chunks[0]["volume_id"]
            volume_ids.add(vol)
            doc_count = len({(row['volume_id'], row['document_id']) for row in chunks})
            stats.documents_chunked += doc_count
            stats.chunks_written += len(chunks)
            for row in chunks:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats.volumes_kept = len(volume_ids)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FRUS 1961+ chunk corpus from HistoryAtState/frus")
    parser.add_argument("--sync", action="store_true", help="Clone/pull the FRUS source repo before parsing")
    parser.add_argument("--chunk-size-words", type=int, default=800)
    parser.add_argument("--overlap-words", type=int, default=100)
    parser.add_argument("--output", type=Path, default=CHUNKS_PATH)
    args = parser.parse_args()

    if args.sync:
        sync_frus_repo()

    if not FRUS_VOLUMES_DIR.exists():
        raise FileNotFoundError(
            f"Missing FRUS volumes directory: {FRUS_VOLUMES_DIR}. Run with --sync first."
        )

    stats = build_corpus(
        output_path=args.output,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.overlap_words,
    )

    print("\nBuild complete")
    print(f"Volumes kept (1961+): {stats.volumes_kept}")
    print(f"Documents chunked: {stats.documents_chunked}")
    print(f"Chunks written: {stats.chunks_written}")
    print(f"Files skipped (parse issues): {stats.files_skipped}")
    print(f"Output JSONL: {args.output}")


if __name__ == "__main__":
    main()
