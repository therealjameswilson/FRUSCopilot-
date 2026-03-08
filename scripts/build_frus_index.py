from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import UTC, datetime

import numpy as np
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    EMBEDDINGS_DB_PATH,
    FRUS_REPO_DIR,
    FRUS_VOLUMES_DIR,
    MANIFEST_PATH,
    OPENAI_API_KEY,
)
from scripts.build_frus_chunks import build_corpus, sync_frus_repo


def load_chunks(path: str) -> list[dict]:
    chunks: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_embeddings(chunks: list[dict]) -> list[tuple[str, bytes]]:
    if not OPENAI_API_KEY:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    results: list[tuple[str, bytes]] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") or chunk.get("id")
        if not chunk_id:
            continue
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk["text"])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        results.append((str(chunk_id), embedding.tobytes()))

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


def build_manifest(chunk_count: int, volume_count: int, embeddings_built: bool) -> dict:
    return {
        "schema_version": "1.2",
        "built_at": datetime.now(UTC).isoformat(),
        "repo_path": str(FRUS_REPO_DIR),
        "volumes_path": str(FRUS_VOLUMES_DIR),
        "chunk_count": chunk_count,
        "volume_count": volume_count,
        "embedding_model": EMBEDDING_MODEL if embeddings_built else None,
        "chunks_path": str(CHUNKS_PATH),
        "embeddings_path": str(EMBEDDINGS_DB_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FRUS chunk index and optional vector index")
    parser.add_argument("--sync", action="store_true", help="Clone/pull FRUS repo first")
    parser.add_argument("--with-embeddings", action="store_true", help="Build OpenAI embeddings and sqlite index")
    parser.add_argument("--chunk-size-words", type=int, default=800)
    parser.add_argument("--overlap-words", type=int, default=100)
    args = parser.parse_args()

    if args.sync:
        sync_frus_repo()

    if not FRUS_VOLUMES_DIR.exists():
        raise FileNotFoundError(
            f"Missing FRUS volumes directory at {FRUS_VOLUMES_DIR}. Run with --sync first."
        )

    stats = build_corpus(
        output_path=CHUNKS_PATH,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.overlap_words,
    )

    chunks = load_chunks(str(CHUNKS_PATH))
    if args.with_embeddings:
        rows = build_embeddings(chunks)
        write_embeddings_sqlite(rows)

    manifest = build_manifest(
        chunk_count=len(chunks),
        volume_count=stats.volumes_kept,
        embeddings_built=args.with_embeddings,
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Built FRUS chunks: {len(chunks)} across {stats.volumes_kept} volumes")
    print(f"Chunks: {CHUNKS_PATH}")
    if args.with_embeddings:
        print(f"Embeddings: {EMBEDDINGS_DB_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
