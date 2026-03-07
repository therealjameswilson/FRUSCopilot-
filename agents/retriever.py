from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from openai import OpenAI

from config import CHUNKS_PATH, EMBEDDING_MODEL, EMBEDDINGS_DB_PATH, OPENAI_API_KEY

TOKEN_RE = re.compile(r"[a-z0-9]+")


class FrusRetriever:
    def __init__(self, chunks_path: Path = CHUNKS_PATH, embeddings_db_path: Path = EMBEDDINGS_DB_PATH):
        self.chunks_path = chunks_path
        self.embeddings_db_path = embeddings_db_path
        self._chunks = self._load_chunks()

    def _load_chunks(self) -> list[dict]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing chunk index: {self.chunks_path}")

        chunks: list[dict] = []
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    @staticmethod
    def _embed_query(query: str) -> np.ndarray:
        if not OPENAI_API_KEY:
            raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _load_embeddings(self) -> dict[str, np.ndarray]:
        if not self.embeddings_db_path.exists():
            raise FileNotFoundError(f"Missing embeddings database: {self.embeddings_db_path}")

        conn = sqlite3.connect(str(self.embeddings_db_path))
        try:
            rows = conn.execute("SELECT chunk_id, embedding FROM embeddings").fetchall()
        finally:
            conn.close()

        parsed: dict[str, np.ndarray] = {}
        for chunk_id, blob in rows:
            parsed[chunk_id] = np.frombuffer(blob, dtype=np.float32)
        return parsed

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        return TOKEN_RE.findall((value or "").lower())

    def _keyword_scores(self, query: str, filters: dict | None = None) -> list[dict]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        filter_volume = (filters or {}).get("volume_slug")

        scored: list[dict] = []
        for chunk in self._chunks:
            if filter_volume and chunk.get("volume_slug") != filter_volume:
                continue

            title = chunk.get("title") or ""
            section = chunk.get("section_title") or ""
            chapter = chunk.get("chapter_title") or ""
            text = chunk.get("text") or ""
            haystack = f"{title} {section} {chapter} {text[:3000]}".lower()
            h_tokens = set(self._tokenize(haystack))
            overlap = len(q_set & h_tokens)
            if overlap == 0:
                continue

            score = overlap / max(len(q_set), 1)
            if query.lower() in haystack:
                score += 0.5
            if any(token in (title + " " + section + " " + chapter).lower() for token in q_set):
                score += 0.25

            scored.append({**chunk, "score": float(score), "retrieval_method": "keyword"})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    def _vector_scores(self, query: str, filters: dict | None = None) -> list[dict]:
        embeddings = self._load_embeddings()
        query_vec = self._embed_query(query)

        filter_volume = (filters or {}).get("volume_slug")

        scored: list[dict] = []
        for chunk in self._chunks:
            if filter_volume and chunk.get("volume_slug") != filter_volume:
                continue

            emb = embeddings.get(chunk["chunk_id"])
            if emb is None:
                continue

            denom = np.linalg.norm(query_vec) * np.linalg.norm(emb)
            score = float(np.dot(query_vec, emb) / denom) if denom else 0.0
            scored.append({**chunk, "score": score, "retrieval_method": "vector"})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    def search(self, query: str, top_k: int = 20, filters: dict | None = None, strategy: str = "hybrid") -> list[dict]:
        if strategy == "keyword":
            return self._keyword_scores(query=query, filters=filters)[:top_k]

        if strategy == "vector":
            return self._vector_scores(query=query, filters=filters)[:top_k]

        keyword = self._keyword_scores(query=query, filters=filters)

        try:
            vector = self._vector_scores(query=query, filters=filters)
        except Exception:
            vector = []

        merged: dict[str, dict] = {}
        for item in keyword + vector:
            key = item.get("chunk_id")
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(item)
            else:
                existing["score"] = max(float(existing.get("score", 0.0)), float(item.get("score", 0.0)))
                methods = set((existing.get("retrieval_method") or "").split("+")) | {
                    item.get("retrieval_method", "")
                }
                existing["retrieval_method"] = "+".join(sorted(m for m in methods if m))

        ranked = list(merged.values())
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]


_DEFAULT_RETRIEVER: FrusRetriever | None = None


def _get_retriever() -> FrusRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = FrusRetriever()
    return _DEFAULT_RETRIEVER


def search(query: str, top_k: int = 20, filters: dict | None = None, strategy: str = "hybrid") -> list[dict]:
    return _get_retriever().search(query=query, top_k=top_k, filters=filters, strategy=strategy)
