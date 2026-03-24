from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document

from src.utils.logger import get_logger

log = get_logger(__name__)


def load_pdfs_from_directory(directory: Path) -> List[Document]:
    directory = directory.resolve()
    if not directory.is_dir():
        log.warning("Documents directory missing: %s", directory)
        return []

    docs: List[Document] = []
    for path in sorted(directory.glob("*.pdf")):
        try:
            with fitz.open(path) as pdf:
                n = pdf.page_count
                for i in range(n):
                    page = pdf[i]
                    body = (page.get_text() or "").strip()
                    if body:
                        docs.append(
                            Document(
                                page_content=body,
                                metadata={
                                    "source": path.name,
                                    "path": str(path),
                                    "page": i + 1,
                                },
                            )
                        )
                log.info("Loaded PDF %s (%s pages with text)", path.name, n)
        except Exception as e:
            log.exception("Failed to load %s: %s", path, e)
    return docs
