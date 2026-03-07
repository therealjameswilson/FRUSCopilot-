from __future__ import annotations

import os
import re
from pathlib import Path
from html.parser import HTMLParser
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from flask import Flask, render_template, request

app = Flask(__name__)
app.config["FRUS_DATA_DIR"] = os.getenv("FRUS_DATA_DIR")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FRUS_DIR = BASE_DIR / "data" / "frus" / "volumes"
MAX_ALLOWED_RESULTS = 200
DEFAULT_RESULT_COUNT = 25
STATUS_PAGE_URL = "https://history.state.gov/historicaldocuments/status-of-the-series"
TARGET_VOLUME_STATUSES = {"being researched", "planner"}
MANUAL_VOLUME_OPTIONS = [
    "Planned — 2001–2008, China Volumes",
    "Planned — Road to 9/11 Volume",
]


class FRUSStatusParser(HTMLParser):
    """Extract table row cell text and links from the FRUS status page."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict[str, str]]] = []
        self._in_tr = False
        self._in_cell = False
        self._current_row: list[dict[str, str]] = []
        self._current_text: list[str] = []
        self._current_link = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "tr":
            self._in_tr = True
            self._current_row = []
        elif self._in_tr and tag in {"td", "th"}:
            self._in_cell = True
            self._current_text = []
            self._current_link = ""
        elif self._in_cell and tag == "a":
            self._current_link = attrs_dict.get("href", "") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            text = " ".join("".join(self._current_text).split())
            self._current_row.append({"text": text, "href": self._current_link})
            self._in_cell = False
        elif tag == "tr" and self._in_tr:
            if self._current_row:
                self.rows.append(self._current_row)
            self._in_tr = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_text.append(data)


def fetch_status_page_html() -> str:
    with urlopen(STATUS_PAGE_URL, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_target_volume_names(status_html: str) -> list[str]:
    parser = FRUSStatusParser()
    parser.feed(status_html)

    volume_names: set[str] = set()
    for row in parser.rows:
        texts = [cell["text"].strip() for cell in row if cell["text"].strip()]
        if not texts:
            continue

        lower_texts = {text.lower() for text in texts}
        if not (lower_texts & TARGET_VOLUME_STATUSES):
            continue

        first_link_text = next((cell["text"].strip() for cell in row if cell["href"] and cell["text"].strip()), "")
        volume_label = first_link_text or texts[0]
        volume_names.add(volume_label)

    return sorted(volume_names)


def normalize_volume_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def doc_matches_volume(doc: dict[str, str], selected_volume: str) -> bool:
    if not selected_volume:
        return True

    selected = normalize_volume_label(selected_volume)
    source = normalize_volume_label(doc.get("source", ""))
    title = normalize_volume_label(doc.get("title", ""))

    selected_index = volume_index_from_label(selected_volume)
    source_index = volume_index_from_label(doc.get("source", ""))

    return (
        selected in source
        or source in selected
        or selected in title
        or (selected_index and selected_index == source_index)
    )




def roman_to_int(value: str) -> int | None:
    numerals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for ch in reversed(value.lower()):
        if ch not in numerals:
            return None
        current = numerals[ch]
        if current < prev:
            total -= current
        else:
            total += current
            prev = current
    return total if total > 0 else None


def volume_index_from_label(label: str) -> str:
    lowered = label.lower()
    roman_match = re.search(r"\bvolume\s+([ivxlcdm]+)\b", lowered)
    if roman_match:
        roman_value = roman_to_int(roman_match.group(1))
        if roman_value is not None:
            return f"v{roman_value}"

    digit_match = re.search(r"(?<![a-z])v(?:ol(?:ume)?)?\s*([0-9]{1,3})\b", lowered)
    if digit_match:
        return f"v{int(digit_match.group(1))}"

    return ""

def get_volume_options() -> tuple[list[str], str | None]:
    try:
        html = fetch_status_page_html()
        options = extract_target_volume_names(html)
    except (URLError, TimeoutError, ValueError) as exc:
        return MANUAL_VOLUME_OPTIONS[:], f"Could not load volume options from {STATUS_PAGE_URL}: {exc}"

    options = sorted(set(options + MANUAL_VOLUME_OPTIONS))

    if not options:
        return [], "No 'Being Researched' or 'Planner' volumes were found on the status page."

    return options, None


def parse_volume(file_path: Path) -> list[dict[str, str]]:
    """Extract FRUS document divisions from one TEI XML volume."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    namespace = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"tei": namespace} if namespace else {}
    query = ".//tei:div" if namespace else ".//div"

    documents: list[dict[str, str]] = []
    for div in root.findall(query, ns):
        if div.get("type") != "document":
            continue

        title_query = "tei:head" if namespace else "head"
        head = div.find(title_query, ns)
        title = " ".join(head.itertext()).strip() if head is not None else "Untitled document"
        text = " ".join(div.itertext()).strip()

        documents.append(
            {
                "title": title,
                "text": text,
                "source": file_path.name,
            }
        )

    return documents


def collect_xml_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.xml"))


def load_documents(folder: Path) -> tuple[list[dict[str, str]], list[str]]:
    documents: list[dict[str, str]] = []
    errors: list[str] = []

    for xml_file in collect_xml_files(folder):
        try:
            documents.extend(parse_volume(xml_file))
        except Exception as exc:  # broad by design to keep UI responsive on malformed files
            errors.append(f"{xml_file.name}: {exc}")

    return documents, errors


def get_data_dir() -> Path:
    configured = app.config.get("FRUS_DATA_DIR")
    if configured:
        return Path(configured)
    return DEFAULT_FRUS_DIR


def parse_max_results(raw_max_results: str | None) -> int:
    try:
        value = int(raw_max_results or DEFAULT_RESULT_COUNT)
    except ValueError:
        return DEFAULT_RESULT_COUNT
    return max(1, min(value, MAX_ALLOWED_RESULTS))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        source_mode = request.form.get("source_mode", "default")
        max_results = parse_max_results(request.form.get("max_results"))
        selected_volume = request.form.get("selected_volume", "").strip()
    else:
        query = request.args.get("query", "").strip()
        source_mode = request.args.get("source_mode", "default")
        max_results = parse_max_results(request.args.get("max_results"))
        selected_volume = request.args.get("selected_volume", "").strip()

    volume_options, volume_error = get_volume_options()
    if selected_volume and selected_volume not in volume_options:
        selected_volume = ""

    docs: list[dict[str, str]] = []
    errors: list[str] = []

    if request.method == "POST" and source_mode == "upload":
        uploaded = request.files.get("xml_file")
        tmp_path = BASE_DIR / ".uploaded_tmp.xml"
        if uploaded and uploaded.filename:
            try:
                uploaded.save(tmp_path)
                docs = parse_volume(tmp_path)
            except Exception as exc:
                errors.append(f"Upload parse error: {exc}")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        else:
            errors.append("Please choose an XML file to upload.")
    else:
        docs, errors = load_documents(get_data_dir())

    filtered_docs = [doc for doc in docs if doc_matches_volume(doc, selected_volume)]
    if query:
        lowered = query.lower()
        filtered_docs = [
            doc
            for doc in filtered_docs
            if lowered in doc["title"].lower() or lowered in doc["text"].lower()
        ]

    stats = {
        "total_documents": len(docs),
        "matching_documents": len(filtered_docs),
        "shown_documents": min(len(filtered_docs), max_results),
    }

    return render_template(
        "index.html",
        query=query,
        source_mode=source_mode,
        max_results=max_results,
        selected_volume=selected_volume,
        volume_options=volume_options,
        docs=filtered_docs[:max_results],
        errors=([volume_error] if volume_error else []) + errors,
        stats=stats,
        data_dir=str(get_data_dir()),
    )


@app.route("/healthz")
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
