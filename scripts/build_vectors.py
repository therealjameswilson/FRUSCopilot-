import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
DOCUMENTS_PATH = DATABASE_DIR / "frus_documents.json"
VECTORS_PATH = DATABASE_DIR / "frus_vectors.npy"
VECTOR_IDS_PATH = DATABASE_DIR / "frus_vector_ids.json"
BATCH_SIZE = 64


def require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "Missing OPENAI_API_KEY environment variable. "
            "Set OPENAI_API_KEY before running scripts/build_vectors.py."
        )


def load_documents(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source documents file: {path}. "
            "You must provide source texts first in database/frus_documents.json."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("database/frus_documents.json must contain a JSON array of objects.")

    validated_docs = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Document at index {index} must be an object with 'id' and 'text'.")

        doc_id = item.get("id")
        text = item.get("text")

        if doc_id is None or text is None:
            raise ValueError(f"Document at index {index} is missing required keys 'id' and/or 'text'.")

        if not isinstance(doc_id, str):
            raise ValueError(f"Document at index {index} has non-string 'id'.")

        if not isinstance(text, str):
            raise ValueError(f"Document at index {index} has non-string 'text'.")

        cleaned_text = text.strip()
        if cleaned_text:
            validated_docs.append({"id": doc_id, "text": cleaned_text})

    if not validated_docs:
        raise ValueError("No non-blank documents found in database/frus_documents.json.")

    return validated_docs


def batched(items: list[dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def build_vectors() -> None:
    require_openai_api_key()
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents(DOCUMENTS_PATH)
    client = OpenAI()

    all_embeddings: list[list[float]] = []
    all_ids: list[str] = []

    for batch in batched(docs, BATCH_SIZE):
        batch_ids = [doc["id"] for doc in batch]
        batch_texts = [doc["text"] for doc in batch]

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch_texts,
        )

        batch_embeddings = [item.embedding for item in response.data]
        if len(batch_embeddings) != len(batch_texts):
            raise RuntimeError("Embedding API returned an unexpected number of vectors.")

        all_ids.extend(batch_ids)
        all_embeddings.extend(batch_embeddings)

    vectors_array = np.array(all_embeddings, dtype=np.float32)

    np.save(VECTORS_PATH, vectors_array)
    with VECTOR_IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_ids, f, ensure_ascii=False, indent=2)

    print(
        "Built vectors successfully: "
        f"documents={len(all_ids)}, shape={vectors_array.shape}, "
        f"vectors='{VECTORS_PATH}', ids='{VECTOR_IDS_PATH}'"
    )


if __name__ == "__main__":
    build_vectors()
