import json
import sqlite3
from pathlib import Path

from agents.retriever import FrusRetriever, get_retrieval_status
import agents.retriever as retriever_module


def _write_chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": "c1",
            "volume_slug": "frus1989-92v01",
            "volume_id": "frus1989-92v01",
            "volume_title": "FRUS 1989-1992 Volume I",
            "title": "Memo for Brent Scowcroft",
            "text": "Scowcroft and NSC process during Bush administration.",
        },
        {
            "chunk_id": "c2",
            "volume_slug": "frus1989-92v02",
            "volume_id": "frus1989-92v02",
            "volume_title": "FRUS 1989-1992 Volume II",
            "title": "NSC staffing",
            "text": "National Security Council staffing memorandum.",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_embeddings(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE embeddings (chunk_id TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
    for cid in ids:
        conn.execute("INSERT INTO embeddings(chunk_id, embedding) VALUES (?, ?)", (cid, b"\x00\x00\x80?" * 8))
    conn.commit()
    conn.close()


def test_keyword_fallback_when_vector_returns_zero(monkeypatch, tmp_path: Path):
    chunks = tmp_path / "chunks.jsonl"
    db = tmp_path / "emb.sqlite"
    _write_chunks(chunks)
    _write_embeddings(db, ["c1", "c2"])

    retriever = FrusRetriever(chunks_path=chunks, embeddings_db_path=db)
    monkeypatch.setattr(retriever, "_vector_scores", lambda query, filters=None: [])

    rows = retriever.search("Scowcroft", strategy="hybrid")
    assert rows
    assert retriever.last_query_details["fallback_used"] is True
    assert retriever.last_query_details["keyword_count"] >= 1


def test_volume_filter_normalization_all_does_not_exclude(tmp_path: Path):
    chunks = tmp_path / "chunks.jsonl"
    db = tmp_path / "emb.sqlite"
    _write_chunks(chunks)
    _write_embeddings(db, ["c1", "c2"])

    retriever = FrusRetriever(chunks_path=chunks, embeddings_db_path=db)

    rows = retriever.search("NSC", strategy="keyword", filters={"volume_slug": "All"})
    assert len(rows) >= 1


def test_get_retrieval_status_includes_alignment(monkeypatch, tmp_path: Path):
    chunks = tmp_path / "chunks.jsonl"
    db = tmp_path / "emb.sqlite"
    _write_chunks(chunks)
    _write_embeddings(db, ["c1"])

    monkeypatch.setattr(retriever_module, "_DEFAULT_RETRIEVER", FrusRetriever(chunks_path=chunks, embeddings_db_path=db))
    status = get_retrieval_status()

    assert status["chunk_count"] == 2
    assert status["embeddings_row_counts"]["embeddings"] == 1
    assert status["chunk_embedding_alignment"]["chunks_missing_embeddings"] == 1
