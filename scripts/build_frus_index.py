from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import UTC, datetime

import numpy as np
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.frus_loader import chunk_document, load_documents, write_chunks_jsonl
from config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    EMBEDDINGS_DB_PATH,
    FRUS_REPO_DIR,
    FRUS_VOLUMES_DIR,
    MANIFEST_PATH,
    OPENAI_API_KEY,
)


def build_embeddings(chunks: list[dict]) -> list[tuple[str, bytes]]:
    if not OPENAI_API_KEY:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    results: list[tuple[str, bytes]] = []

    for chunk in chunks:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk["text"])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        results.append((chunk["chunk_id"], embedding.tobytes()))

    return results


def write_embeddings_sqlite(rows: list[tuple[str, bytes]]) -> None:
    EMBEDDINGS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(EMBEDDINGS_DB_PATH))
    try:
        conn.execute("DROP TABLE IF EXISTS embeddings")
        conn.execute("CREATE TABLE embeddings (chunk_id TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
        conn.executemany("INSERT INTO embeddings(chunk_id, embedding) VALUES (?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def build_manifest(chunk_count: int, volume_count: int) -> dict:
    return {
        "schema_version": "1.1",
        "built_at": datetime.now(UTC).isoformat(),
        "repo_path": str(FRUS_REPO_DIR),
        "volumes_path": str(FRUS_VOLUMES_DIR),
        "chunk_count": chunk_count,
        "volume_count": volume_count,
        "embedding_model": EMBEDDING_MODEL,
        "chunks_path": str(CHUNKS_PATH),
        "embeddings_path": str(EMBEDDINGS_DB_PATH),
    }


def main() -> None:
    if not FRUS_VOLUMES_DIR.exists():
        raise FileNotFoundError(
            f"Missing FRUS volumes directory at {FRUS_VOLUMES_DIR}. Run scripts/sync_frus_repo.py first."
        )

    chunks: list[dict] = []
    volumes: set[str] = set()

    for doc in load_documents(FRUS_VOLUMES_DIR, FRUS_REPO_DIR):
        volumes.add(doc.volume_slug)
        chunks.extend(chunk_document(doc))

    write_chunks_jsonl(chunks, CHUNKS_PATH)
    embedding_rows = build_embeddings(chunks)
    write_embeddings_sqlite(embedding_rows)

    manifest = build_manifest(chunk_count=len(chunks), volume_count=len(volumes))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Built FRUS index with {len(chunks)} chunks across {len(volumes)} volumes.")
    print(f"Chunks: {CHUNKS_PATH}")
    print(f"Embeddings: {EMBEDDINGS_DB_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
