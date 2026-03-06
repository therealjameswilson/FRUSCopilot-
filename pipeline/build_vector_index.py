import os
import xml.etree.ElementTree as ET
from openai import OpenAI
import numpy as np

FRUS_DIR = "data/frus/volumes"

client = OpenAI()

def extract_documents(file_path):

    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = {'tei': root.tag.split('}')[0].strip('{')}

    docs = []

    for div in root.findall(".//tei:div", ns):

        if div.get("type") == "document":

            text = "".join(div.itertext())

            docs.append(text[:4000])  # truncate long docs

    return docs


def embed_text(text):

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )

    return response.data[0].embedding


def process_volumes():

    vectors = []

    for root_dir, dirs, files in os.walk(FRUS_DIR):

        for file in files:

            if file.endswith(".xml"):

                path = os.path.join(root_dir, file)

                documents = extract_documents(path)

                for doc in documents:

                    vector = embed_text(doc)

                    vectors.append(vector)

                print(file, "embedded:", len(documents))

    np.save("database/frus_vectors.npy", vectors)


if __name__ == "__main__":
    process_volumes()
