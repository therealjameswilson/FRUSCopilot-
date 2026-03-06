import os
import xml.etree.ElementTree as ET

FRUS_DIR = "data/frus/volumes"

def extract_documents(file_path):

    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = {'tei': root.tag.split('}')[0].strip('{')}

    documents = []

    for div in root.findall(".//tei:div", ns):

        if div.get("type") == "document":

            doc = {}

            doc["text"] = "".join(div.itertext())

            head = div.find("tei:head", ns)
            if head is not None:
                doc["title"] = "".join(head.itertext())

            documents.append(doc)

    return documents


def process_volumes():

    for root_dir, dirs, files in os.walk(FRUS_DIR):

        for file in files:

            if file.endswith(".xml"):

                path = os.path.join(root_dir, file)

                docs = extract_documents(path)

                print(file, "extracted docs:", len(docs))


if __name__ == "__main__":
    process_volumes()
