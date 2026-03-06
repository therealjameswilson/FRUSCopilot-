import os
import xml.etree.ElementTree as ET

FRUS_DIR = "data/frus/volumes"

def parse_volume(file_path):

    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract namespace automatically
    ns = {'tei': root.tag.split('}')[0].strip('{')}

    documents = []

    for div in root.findall(".//tei:div", ns):

        if div.get("type") == "document":

            text = "".join(div.itertext())
            documents.append(text)

    return documents


def process_volumes():

    for root_dir, dirs, files in os.walk(FRUS_DIR):

        for file in files:

            if file.endswith(".xml"):

                path = os.path.join(root_dir, file)

                try:
                    docs = parse_volume(path)
                    print(file, "documents:", len(docs))

                except Exception as e:
                    print("Error reading:", file, e)


if __name__ == "__main__":
    process_volumes()
