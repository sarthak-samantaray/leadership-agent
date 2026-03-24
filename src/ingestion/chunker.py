from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import LeadershipAgentConfig


def chunk_documents(documents: List[Document], cfg: LeadershipAgentConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag.chunk_size,
        chunk_overlap=cfg.rag.chunk_overlap,
        length_function=len,
    )
    out: List[Document] = []
    for doc in documents:
        for i, chunk in enumerate(splitter.split_documents([doc])):
            chunk.metadata = {**chunk.metadata, "chunk_index": i}
            out.append(chunk)
    return out
