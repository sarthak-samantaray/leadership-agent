"""Cohere cross-encoder rerank for RAG (Rerank API)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document

log = get_logger(__name__)


def cohere_rerank_documents(
    query: str,
    documents: List["Document"],
    *,
    api_key: str,
    model: str,
    top_n: int,
) -> List["Document"]:
    """Rerank LangChain Documents by semantic relevance using Cohere Rerank."""
    if not documents:
        return documents
    if not api_key.strip():
        return documents
    if len(documents) == 1:
        return documents

    try:
        import cohere
    except ImportError as e:
        log.warning("cohere package missing; install cohere: %s", e)
        return documents

    texts = [d.page_content or "" for d in documents]
    n = max(1, min(top_n, len(texts)))
    client = cohere.Client(api_key=api_key.strip())
    try:
        resp = client.rerank(
            model=model,
            query=query,
            documents=texts,
            top_n=n,
        )
    except Exception as e:
        log.exception("Cohere rerank failed: %s", e)
        return documents[:n]

    results = getattr(resp, "results", None) or []
    out: List["Document"] = []
    for r in results:
        idx = getattr(r, "index", None)
        if idx is None and isinstance(r, dict):
            idx = r.get("index")
        if idx is None:
            continue
        if 0 <= int(idx) < len(documents):
            out.append(documents[int(idx)])
    return out if out else documents[:n]
