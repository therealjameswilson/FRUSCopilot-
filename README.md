# FRUS Copilot Web App

This repository includes a lightweight public web interface that lets colleagues search FRUS TEI XML files from a browser.

## What it does

- Loads XML files from `data/frus/volumes` (recursive) and extracts `<div type="document">` sections.
- Supports keyword search over title and full document text.
- Optionally allows uploading a single XML file directly from the page.
- Supports shareable URL queries (GET) and a health endpoint at `/healthz` for uptime checks.


## Build the local FRUS 1961+ corpus

This app now retrieves from a local JSONL chunk corpus built from `HistoryAtState/frus` TEI/XML volumes.

1. Build/sync and chunk the corpus:

```bash
python3 scripts/build_frus_chunks.py --sync
```

Output: `data/index/frus_chunks_1961_plus.jsonl` (1961+ volumes only).

2. (Optional) Build embeddings for vector retrieval:

```bash
python3 scripts/build_frus_index.py --sync --with-embeddings
```

The chunk schema includes metadata fields used by retrieval and future compiler-assist reranking, including: `id`, `text`, `volume_id`, `volume_title`, `volume_start_year`, `volume_end_year`, `document_id`, `document_title`, `chapter_title`, `compilation_title`, `section_title`, `source_url`, `source_file`, `chunk_index`, and when available `administration` and `doc_date`.

Canonical FRUS URLs are mapped as:
`https://history.state.gov/historicaldocuments/{volume_id}/{document_id}`

The Streamlit app (`ui/app.py`) consumes `data/index/frus_chunks_1961_plus.jsonl` via `agents/retriever.py` and now differentiates these states:
- corpus file missing
- corpus file empty
- corpus load failure
- no search hits for the current query/filter

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`.

## Compiler Assist (Inference Mode)

The Streamlit app now supports two retrieval modes:

- **Exact Retrieval**: regular FRUS retrieval behavior.
- **Compiler Assist (Inference Mode)**: inference/planning + multi-query retrieval + reranking + a structured compiler brief.

Compiler Assist keeps retrieval scoped to FRUS-approved corpora only (history.state.gov FRUS URLs represented in the local chunk index and chunked content from `HistoryAtState/frus`). If literal matches are sparse, it transparently falls back to inference-expanded FRUS retrieval and explains that the exact phrase was missing.

Run the Streamlit UI:

```bash
streamlit run ui/app.py
```

## Public share link from this repo

A static, browser-only version of the FRUS interface is available at `docs/index.html`.

- GitHub file URL (works immediately): `https://github.com/<org>/<repo>/blob/<branch>/docs/index.html`
- GitHub Pages URL (after enabling Pages for `/docs`): `https://<org>.github.io/<repo>/`

To enable GitHub Pages:
1. Go to **Settings → Pages** in your repo.
2. Set **Source** to **Deploy from a branch**.
3. Select your branch and **/docs** folder.
4. Save and share the generated Pages URL.

## Deploy publicly

This app is ready to run behind Gunicorn on any VM/PaaS.

```bash
gunicorn --bind 0.0.0.0:8000 app:app
```

### Recommended environment variable

- `FRUS_DATA_DIR`: absolute or relative path to your shared FRUS XML volume directory.

Example:

```bash
FRUS_DATA_DIR=/srv/frus/volumes gunicorn --bind 0.0.0.0:8000 app:app
```


### Streamlit Community Cloud secret

If you deploy the Streamlit app, set `OPENAI_API_KEY` in **App settings → Secrets** (or environment variables) so OpenAI client initialization works at runtime.

## Notes

- The app displays parse warnings for malformed XML files rather than crashing.
- If no local FRUS files are present, use the upload option to test parsing.
- Use a reverse proxy (Nginx/Caddy/Cloudflare Tunnel) plus HTTPS for internet-facing deployment.
