import os
import re

from llama_index.core import Document


def extract_metadata(path: str) -> dict:
    path_parts = path.strip(os.sep).split(os.sep)

    if len(path_parts) < 3:
        return {}

    ort = path_parts[-3]
    typ = path_parts[-2]

    filename = os.path.basename(path)

    match = re.search(r'(\d{4})', filename)

    jahr = match.group(1) if match else None

    return {
        "typ": typ,
        "ort": ort,
        "jahr": jahr
    }


def extract_pages(docs: list[Document]) -> list[Document]:
    new_docs = []

    # Regex-Muster für Trennzeichen wie {1}-------- oder {17}-----
    split_regex = r"\{\d+\}-+"

    for doc in docs:
        # Auftrennen anhand des Musters
        pages = re.split(split_regex, doc.text)

        # Entferne leere Seiten & erstelle neue Dokumente 
        for i, page in enumerate(pages):
            page = page.strip()
            if not page:
                continue

            new_doc = Document(
                text=page,
                metadata={**doc.metadata, "seite": i + 1}
            )
            new_docs.append(new_doc)

    return new_docs
