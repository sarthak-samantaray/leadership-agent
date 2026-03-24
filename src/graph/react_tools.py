"""ReAct tools: semantic RAG + Tavily web search."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.tools import tool
from tavily import TavilyClient

from src.utils.config import LeadershipAgentConfig
from src.utils.logger import get_logger
from src.vectorstore.cohere_rerank import cohere_rerank_documents
from src.vectorstore.store_manager import load_vectorstore

log = get_logger(__name__)


def make_react_tools(cfg: LeadershipAgentConfig, project_root: Path) -> List:
    _ = project_root

    @tool
    def rag_search(query: str) -> str:
        """Search internal company documents (reports, strategy, ops) via semantic retrieval + optional Cohere rerank.
        Use for performance, financials, strategy, operations, risks. Returns grounded passages with source paths."""
        print("[tool] rag_search", flush=True)
        try:
            vs = load_vectorstore(cfg)
        except FileNotFoundError:
            return "ERROR: No vector store. Run: python ingest.py"
        try:
            cohere_key = cfg.cohere_rerank_api_key()
            use_rerank = cfg.rag.cohere.enabled and bool(cohere_key)
            if cfg.rag.cohere.enabled and not cohere_key.strip():
                log.warning(
                    "Cohere rerank enabled but no key (%s); skipping rerank",
                    cfg.rag.cohere.api_key_env,
                )
            if use_rerank:
                docs = vs.max_marginal_relevance_search(
                    query,
                    k=cfg.rag.mmr_fetch_k,
                    fetch_k=max(cfg.rag.mmr_fetch_k * 2, 40),
                )
                print("[tool] cohere_rerank", flush=True)
                docs = cohere_rerank_documents(
                    query,
                    docs,
                    api_key=cohere_key,
                    model=cfg.rag.cohere.model,
                    top_n=cfg.rag.mmr_k,
                )
            else:
                docs = vs.max_marginal_relevance_search(
                    query,
                    k=cfg.rag.mmr_k,
                    fetch_k=cfg.rag.mmr_fetch_k,
                )
        except Exception as e:
            log.exception("rag_search failed: %s", e)
            return f"RAG search error: {e}"
        if not docs:
            return "No relevant documents found in internal knowledge base."
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            if page is not None:
                label = f"{src} · page {page}"
            else:
                label = str(src)
            parts.append(f"[SOURCE {i}: {label}]\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    @tool
    def web_search(query: str) -> str:
        """Search the web for current market data, news, benchmarks, or context not in internal documents."""
        print("[tool] web_search", flush=True)
        key = cfg.web_search_api_key()
        if not cfg.web_search.enabled or not key:
            return "Web search is disabled or API key missing (set TAVILY / config)."
        client = TavilyClient(api_key=key)
        try:
            response = client.search(
                query=query,
                max_results=cfg.web_search.max_results,
                search_depth="advanced",
                include_answer=True,
            )
        except Exception as e:
            log.exception("web_search failed: %s", e)
            return f"Web search error: {e}"
        results = response.get("results") or []
        if not results:
            return "No web results found."
        lines: list[str] = []
        ans = response.get("answer") or ""
        if ans:
            lines.append(ans)
            lines.append("")
        for i, r in enumerate(results[: cfg.web_search.max_results], 1):
            url = r.get("url", "unknown")
            title = r.get("title", "")
            content = (r.get("content") or "")[:600]
            lines.append(f"[WEB SOURCE {i}: {url}]")
            lines.append(f"Title: {title}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip()

    return [rag_search, web_search]
