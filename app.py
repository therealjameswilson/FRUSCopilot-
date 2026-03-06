from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from flask import Flask, render_template, request

app = Flask(__name__)
app.config["FRUS_DATA_DIR"] = os.getenv("FRUS_DATA_DIR")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FRUS_DIR = BASE_DIR / "data" / "frus" / "volumes"
MAX_ALLOWED_RESULTS = 200
DEFAULT_RESULT_COUNT = 25


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
    else:
        query = request.args.get("query", "").strip()
        source_mode = request.args.get("source_mode", "default")
        max_results = parse_max_results(request.args.get("max_results"))

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

    filtered_docs = docs
    if query:
        lowered = query.lower()
        filtered_docs = [
            doc for doc in docs if lowered in doc["title"].lower() or lowered in doc["text"].lower()
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
        docs=filtered_docs[:max_results],
        errors=errors,
        stats=stats,
        data_dir=str(get_data_dir()),
    )


@app.route("/healthz")
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
