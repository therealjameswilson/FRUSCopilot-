from __future__ import annotations

from pathlib import Path
from typing import Iterable

from flask import Flask, render_template, request

from pipeline.extract_documents import extract_documents

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "frus" / "volumes"

app = Flask(__name__)


def discover_xml_files(limit: int = 50) -> list[str]:
    if not DATA_DIR.exists():
        return []
    files: Iterable[Path] = DATA_DIR.rglob("*.xml")
    return sorted(str(path.relative_to(BASE_DIR)) for path in files)[:limit]


@app.get("/")
def index():
    return render_template(
        "index.html",
        xml_files=discover_xml_files(),
        selected_file="",
        documents=[],
        error_message="",
    )


@app.post("/")
def parse_file():
    selected_file = request.form.get("file_path", "").strip()
    xml_files = discover_xml_files()
    documents = []
    error_message = ""

    if not selected_file:
        error_message = "Please enter an XML file path."
    else:
        target_file = (BASE_DIR / selected_file).resolve()
        try:
            target_file.relative_to(BASE_DIR)
        except ValueError:
            error_message = "Path must be inside the repository."
        else:
            if not target_file.exists() or target_file.suffix.lower() != ".xml":
                error_message = "File does not exist or is not an XML file."
            else:
                try:
                    documents = extract_documents(str(target_file))
                except Exception as exc:  # broad exception keeps UI resilient for malformed files
                    error_message = f"Could not parse XML file: {exc}"

    return render_template(
        "index.html",
        xml_files=xml_files,
        selected_file=selected_file,
        documents=documents,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
