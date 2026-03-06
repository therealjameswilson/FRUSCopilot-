# FRUS Copilot Web App

This repository now includes a lightweight website that lets colleagues search FRUS TEI XML files from a browser.

## What it does

- Loads XML files from `data/frus/volumes` (recursive) and extracts `<div type="document">` sections.
- Supports keyword search over title and full document text.
- Optionally allows uploading a single XML file directly from the page.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`.

## Notes

- The app displays parse warnings for malformed XML files rather than crashing.
- If no local FRUS files are present, use the upload option to test parsing.
