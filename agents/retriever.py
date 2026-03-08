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
        self._chunks: list[dict] = []
        self.last_query_details: dict[str, object] = {}
        self.last_status: str = "uninitialized"
        self.last_error: str | None = None
        self._load_chunks()

    def _load_chunks(self) -> list[dict]:
        self.last_error = None
        if not self.chunks_path.exists():
            self.last_status = "missing_corpus"
            self._chunks = []
            return self._chunks

        chunks: list[dict] = []
        try:
            with self.chunks_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        except Exception as exc:
            self.last_status = "load_failed"
            self.last_error = str(exc)
            self._chunks = []
            return self._chunks

        self._chunks = chunks
        self.last_status = "ready" if chunks else "empty_corpus"
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

    @staticmethod
    def _chunk_key(chunk: dict) -> str:
        return str(chunk.get("chunk_id") or chunk.get("id") or "")

    @staticmethod
    def _chunk_volume(chunk: dict) -> str:
        return str(chunk.get("volume_slug") or chunk.get("volume_id") or "")

    @staticmethod
    def _normalize_value(value: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", "", (value or "").lower())

    def _normalize_volume_filter(self, filters: dict | None) -> str | None:
        raw_value = str((filters or {}).get("volume_slug") or "").strip()
        if not raw_value:
            return None
        lowered = raw_value.lower()
        if lowered in {"all", "any", "*", "none", "(all)", "no filter"}:
            return None
        return raw_value

    def _chunk_matches_volume_filter(self, chunk: dict, volume_filter: str | None) -> bool:
        if not volume_filter:
            return True

        filter_norm = self._normalize_value(volume_filter)
        candidates = [
            str(chunk.get("volume_slug") or ""),
            str(chunk.get("volume_id") or ""),
            str(chunk.get("volume_title") or ""),
        ]
        candidate_norms = [self._normalize_value(value) for value in candidates if value]
        return any(filter_norm == value or filter_norm in value for value in candidate_norms)

    def _filtered_chunks(self, filters: dict | None = None) -> list[dict]:
        volume_filter = self._normalize_volume_filter(filters)
        return [chunk for chunk in self._chunks if self._chunk_matches_volume_filter(chunk, volume_filter)]

    def _keyword_scores(self, query: str, filters: dict | None = None) -> list[dict]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        filtered_chunks = self._filtered_chunks(filters)

        scored: list[dict] = []
        for chunk in filtered_chunks:

            title = chunk.get("title") or chunk.get("document_title") or ""
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
        filtered_chunks = self._filtered_chunks(filters)

        scored: list[dict] = []
        for chunk in filtered_chunks:

            key = self._chunk_key(chunk)
            emb = embeddings.get(key)
            if emb is None:
                continue

            denom = np.linalg.norm(query_vec) * np.linalg.norm(emb)
            score = float(np.dot(query_vec, emb) / denom) if denom else 0.0
            scored.append({**chunk, "score": score, "retrieval_method": "vector"})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    def search(self, query: str, top_k: int = 20, filters: dict | None = None, strategy: str = "hybrid") -> list[dict]:
        if self.last_status in {"missing_corpus", "empty_corpus", "load_failed"}:
            self.last_query_details = {
                "query": query,
                "strategy": strategy,
                "filter_volume": self._normalize_volume_filter(filters),
                "filtered_candidate_count": 0,
                "keyword_count": 0,
                "vector_count": 0,
                "fallback_used": False,
            }
            return []

        filtered_chunks = self._filtered_chunks(filters)
        query_details = {
            "query": query,
            "strategy": strategy,
            "filter_volume": self._normalize_volume_filter(filters),
            "filtered_candidate_count": len(filtered_chunks),
            "keyword_count": 0,
            "vector_count": 0,
            "fallback_used": False,
        }

        if strategy == "keyword":
            keyword_only = self._keyword_scores(query=query, filters=filters)[:top_k]
            query_details["keyword_count"] = len(keyword_only)
            self.last_query_details = query_details
            return keyword_only

        if strategy == "vector":
            vector_only = self._vector_scores(query=query, filters=filters)[:top_k]
            query_details["vector_count"] = len(vector_only)
            if not vector_only:
                keyword_fallback = self._keyword_scores(query=query, filters=filters)[:top_k]
                for item in keyword_fallback:
                    item["retrieval_method"] = "keyword_fallback"
                query_details["keyword_count"] = len(keyword_fallback)
                query_details["fallback_used"] = True
                self.last_query_details = query_details
                return keyword_fallback
            self.last_query_details = query_details
            return vector_only

        keyword = self._keyword_scores(query=query, filters=filters)
        query_details["keyword_count"] = len(keyword)

        try:
            vector = self._vector_scores(query=query, filters=filters)
        except Exception:
            vector = []
        query_details["vector_count"] = len(vector)

        if not vector and keyword:
            query_details["fallback_used"] = True

        merged: dict[str, dict] = {}
        for item in keyword + vector:
            key = self._chunk_key(item)
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
        self.last_query_details = query_details
        return ranked[:top_k]


_DEFAULT_RETRIEVER: FrusRetriever | None = None


def _get_retriever() -> FrusRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = FrusRetriever()
    return _DEFAULT_RETRIEVER


def search(query: str, top_k: int = 20, filters: dict | None = None, strategy: str = "hybrid") -> list[dict]:
    return _get_retriever().search(query=query, top_k=top_k, filters=filters, strategy=strategy)


def get_retrieval_status() -> dict[str, object]:
    retriever = _get_retriever()

    chunks_exists = retriever.chunks_path.exists()
    embeddings_db_exists = retriever.embeddings_db_path.exists()

    chunk_line_count = 0
    if chunks_exists:
        with retriever.chunks_path.open("r", encoding="utf-8") as handle:
            chunk_line_count = sum(1 for line in handle if line.strip())

    embeddings_tables: list[str] = []
    embeddings_row_counts: dict[str, int] = {}
    if embeddings_db_exists:
        conn = sqlite3.connect(str(retriever.embeddings_db_path))
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            embeddings_tables = [str(row[0]) for row in rows]
            for table in embeddings_tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                embeddings_row_counts[table] = int(count)
        finally:
            conn.close()

    chunk_ids = {retriever._chunk_key(chunk) for chunk in retriever._chunks if retriever._chunk_key(chunk)}
    embedding_ids: set[str] = set()
    if embeddings_db_exists and "embeddings" in embeddings_tables:
        conn = sqlite3.connect(str(retriever.embeddings_db_path))
        try:
            rows = conn.execute("SELECT chunk_id FROM embeddings").fetchall()
            embedding_ids = {str(row[0]) for row in rows if row and row[0] is not None}
        finally:
            conn.close()

    distinct_volumes = sorted(
        {
            retriever._chunk_volume(chunk)
            for chunk in retriever._chunks
            if retriever._chunk_volume(chunk)
        }
    )

    return {
        "status": retriever.last_status,
        "error": retriever.last_error,
        "chunks_path": str(retriever.chunks_path),
        "chunks_exists": chunks_exists,
        "chunk_count": len(retriever._chunks),
        "chunk_line_count": chunk_line_count,
        "distinct_volume_count": len(distinct_volumes),
        "distinct_volumes_sample": distinct_volumes[:20],
        "embeddings_db_path": str(retriever.embeddings_db_path),
        "embeddings_db_exists": embeddings_db_exists,
        "embeddings_tables": embeddings_tables,
        "embeddings_row_counts": embeddings_row_counts,
        "chunk_embedding_alignment": {
            "chunk_id_count": len(chunk_ids),
            "embedding_id_count": len(embedding_ids),
            "matching_ids": len(chunk_ids & embedding_ids),
            "chunks_missing_embeddings": len(chunk_ids - embedding_ids),
            "embeddings_missing_chunks": len(embedding_ids - chunk_ids),
        },
        "last_query_details": retriever.last_query_details,
    }
