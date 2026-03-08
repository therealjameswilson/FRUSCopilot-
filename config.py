from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "index"
FRUS_REPO_DIR = CACHE_DIR / "frus_repo"
FRUS_VOLUMES_DIR = FRUS_REPO_DIR / "volumes"

CHUNKS_PATH = INDEX_DIR / "frus_chunks_1961_plus.jsonl"
EMBEDDINGS_DB_PATH = INDEX_DIR / "embeddings.sqlite"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

FRUS_GIT_URL = "https://github.com/HistoryAtState/frus"

EMBEDDING_MODEL = os.getenv("FRUS_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_CHARS_PER_CHUNK = int(os.getenv("FRUS_MAX_CHARS_PER_CHUNK", "12000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("FRUS_CHUNK_OVERLAP_CHARS", "300"))
