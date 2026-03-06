# FRUS Copilot

This repository now includes a simple web interface for exploring FRUS XML files.

## Quick start (web interface)

1. Create/activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   python app.py
   ```
4. Open <http://localhost:8000>.

## How to use the UI

- Pick a detected XML file from the dropdown (when files exist under `data/frus/volumes`).
- Or type a repository-relative XML path manually.
- Click **Parse XML** to extract `div[type="document"]` nodes and view previews.

## Existing pipeline scripts

- `pipeline/parse_frus_xml.py` : batch parsing/counting of document nodes.
- `pipeline/extract_documents.py` : extracts document text and title metadata.
