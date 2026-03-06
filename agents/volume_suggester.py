import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORS_PATH = BASE_DIR / "database" / "frus_vectors.npy"
VECTOR_IDS_PATH = BASE_DIR / "database" / "frus_vector_ids.json"

if not VECTORS_PATH.exists():
    raise FileNotFoundError(f"Missing vector file: {VECTORS_PATH}")

vectors = np.load(VECTORS_PATH, allow_pickle=True)

vector_ids = None
if VECTOR_IDS_PATH.exists():
    with VECTOR_IDS_PATH.open("r", encoding="utf-8") as f:
        loaded_ids = json.load(f)
    if isinstance(loaded_ids, list) and len(loaded_ids) == len(vectors):
        vector_ids = loaded_ids


def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return np.array(response.data[0].embedding)


def suggest_documents(topic, top_k=20):
    query_vec = embed_query(topic)
    scores = []

    for i, vec in enumerate(vectors):
        score = np.dot(query_vec, vec)
        doc_ref = vector_ids[i] if vector_ids is not None else i
        scores.append((doc_ref, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def suggest_declassified_sources(topic):
    prompt = f"""
You are assisting historians compiling a FRUS volume.

Suggest online sources containing declassified or
born-unclassified U.S. government documents relevant to:

{topic}

Prioritize:

• National Archives (NARA)
• CIA FOIA Electronic Reading Room
• State Department FOIA releases
• Presidential Library digital archives
• Defense Department FOIA releases
• Congressional hearings
• Government Publishing Office

Return a short list.
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return response.output_text


def suggest_classified_archives(topic):
    prompt = f"""
You are assisting a FRUS historian.

If documents are not available online, suggest likely
U.S. government archival collections that may contain
still-classified or recently declassified material.

Topic:

{topic}

Focus on:

• Presidential Library collections
• National Security Council files
• Department of State lot files
• Defense Department archives
• CIA operational files
• Joint Chiefs of Staff files

Return likely collections to search.
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return response.output_text


def main():
    print("\nFRUS Volume Builder")
    print("-------------------")

    topic = input("\nEnter proposed FRUS volume topic: ")

    results = suggest_documents(topic)

    print("\nSuggested FRUS documents:\n")

    for r in results:
        print("Doc ID:", r[0], "| relevance:", round(r[1], 3))

    print("\nDeclassified sources to search:\n")
    print(suggest_declassified_sources(topic))

    print("\nPossible classified archival collections:\n")
    print(suggest_classified_archives(topic))


if __name__ == "__main__":
    main()
