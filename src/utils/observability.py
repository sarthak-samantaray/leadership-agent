"""Langfuse tracing for LangGraph / LangChain (SDK reads keys from environment)."""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.utils.config import LeadershipAgentConfig


def build_langfuse_callbacks(cfg: LeadershipAgentConfig) -> List[Any]:
    lf = cfg.langfuse
    if not lf.enabled:
        return []

    sk = os.environ.get("LANGFUSE_SECRET_KEY") or lf.secret_key
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY") or lf.public_key
    base = (
        os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST")
        or lf.base_url
        or ""
    ).rstrip("/")

    if not sk or not pk:
        return []

    base = base or lf.base_url or "https://us.cloud.langfuse.com"
    os.environ["LANGFUSE_BASE_URL"] = base
    os.environ["LANGFUSE_SECRET_KEY"] = sk
    os.environ["LANGFUSE_PUBLIC_KEY"] = pk

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(public_key=pk, secret_key=sk, base_url=base)
    return [CallbackHandler(public_key=pk)]
