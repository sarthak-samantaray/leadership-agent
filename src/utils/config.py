"""Load leadership_agent_config.json (project-local; not application_conf)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.utils.logger import get_logger

_log = get_logger(__name__)


def _looks_like_environment_variable_name(name: str) -> bool:
    """POSIX-style NAME; Cohere keys often fail this (e.g. start with a digit)."""
    if not name or len(name) > 64:
        return False
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


class PathsConfig(BaseModel):
    documents_dir: str = "data/documents"
    vectorstore_dir: str = "vectorstore_db"
    memory_db: str = "memory_db/checkpoints.db"
    reports_dir: str = "reports"
    unified_report_template: str = "templates/unified_report.html"
    guardrails_config: str = "guardrails_config/config.yml"


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str
    model: str = "gemini-2.5-flash-lite"
    timeout_seconds: int = 80
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 0.95


class EmbeddingsConfig(BaseModel):
    """Local sentence-transformers model (HuggingFace) for Chroma."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"


class WebSearchConfig(BaseModel):
    enabled: bool = False
    provider: str = "tavily"
    api_key: str = ""
    api_key_env: str = "TAVILY_API_KEY"
    max_results: int = 5


class CohereRerankConfig(BaseModel):
    """Cohere Rerank API (cross-encoder) after vector retrieval."""

    enabled: bool = False
    model: str = "rerank-english-v3.0"
    api_key: str = ""
    api_key_env: str = "COHERE_API_KEY"
    # Final number of chunks after rerank (defaults to mmr_k from parent RAGConfig at call site)


class RAGConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 100
    mmr_k: int = 5
    mmr_fetch_k: int = 20
    cohere: CohereRerankConfig = Field(default_factory=CohereRerankConfig)


class LangfuseConfig(BaseModel):
    enabled: bool = True
    secret_key: str = ""
    public_key: str = ""
    base_url: str = "https://us.cloud.langfuse.com"


class ReactAgentConfig(BaseModel):
    """ReAct loop cap: ~max_tool_rounds model↔tool cycles; recursion_limit overrides graph steps."""

    max_tool_rounds: int = 5
    recursion_limit: Optional[int] = None

    def effective_recursion_limit(self) -> int:
        if self.recursion_limit is not None:
            return max(4, min(self.recursion_limit, 64))
        # Room for: model↔tools cycles (RAG, web, mermaid, report) + final reply
        return min(64, 2 * self.max_tool_rounds + 14)


class LeadershipAgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    react_agent: ReactAgentConfig = Field(default_factory=ReactAgentConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> LeadershipAgentConfig:
        root = Path(__file__).resolve().parents[2]
        cfg_path = path or (root / "leadership_agent_config.json")
        raw: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
        clean = {k: v for k, v in raw.items() if not k.startswith("_")}
        if "paths" in clean and isinstance(clean["paths"], dict):
            clean["paths"] = {k: v for k, v in clean["paths"].items() if not str(k).startswith("_")}
        if "llm" in clean and isinstance(clean["llm"], dict):
            clean["llm"] = {k: v for k, v in clean["llm"].items() if not str(k).startswith("_")}
        if "embeddings" in clean and isinstance(clean["embeddings"], dict):
            clean["embeddings"] = {
                k: v for k, v in clean["embeddings"].items() if not str(k).startswith("_")
            }
        if "web_search" in clean and isinstance(clean["web_search"], dict):
            clean["web_search"] = {
                k: v for k, v in clean["web_search"].items() if not str(k).startswith("_")
            }
        if "rag" in clean and isinstance(clean["rag"], dict):
            rag = {k: v for k, v in clean["rag"].items() if not str(k).startswith("_")}
            if "cohere" in rag and isinstance(rag["cohere"], dict):
                rag["cohere"] = {
                    k: v for k, v in rag["cohere"].items() if not str(k).startswith("_")
                }
            clean["rag"] = rag
        if "langfuse" in clean and isinstance(clean["langfuse"], dict):
            clean["langfuse"] = {
                k: v for k, v in clean["langfuse"].items() if not str(k).startswith("_")
            }
            if "host" in clean["langfuse"] and "base_url" not in clean["langfuse"]:
                clean["langfuse"]["base_url"] = clean["langfuse"].pop("host")
        if "react_agent" in clean and isinstance(clean["react_agent"], dict):
            clean["react_agent"] = {
                k: v for k, v in clean["react_agent"].items() if not str(k).startswith("_")
            }
        return cls.model_validate(clean)

    def resolve(self, relative: str) -> Path:
        root = Path(__file__).resolve().parents[2]
        return (root / relative).resolve()

    def web_search_api_key(self) -> str:
        if self.web_search.api_key:
            return self.web_search.api_key
        return os.environ.get(self.web_search.api_key_env, "")

    def cohere_rerank_api_key(self) -> str:
        """Prefer rag.cohere.api_key; else env rag.cohere.api_key_env.

        If the secret was mistakenly put in api_key_env (common), use it as the key
        and log a hint — api_key_env must be a name like COHERE_API_KEY, not the token.
        """
        c = self.rag.cohere
        if (c.api_key or "").strip():
            return c.api_key.strip()
        raw_env = (c.api_key_env or "").strip()
        if not raw_env:
            return ""
        if _looks_like_environment_variable_name(raw_env):
            return os.environ.get(raw_env, "")
        _log.warning(
            "rag.cohere.api_key is empty but api_key_env does not look like an env var "
            "name (expected e.g. COHERE_API_KEY). Using api_key_env value as the API key; "
            "move it to rag.cohere.api_key and set api_key_env to COHERE_API_KEY."
        )
        return raw_env
