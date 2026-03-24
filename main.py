#!/usr/bin/env python3
"""Interactive CLI for the Adobe Leadership agent (simple ReAct + RAG + web)."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime_bootstrap import apply_runtime_bootstrap

apply_runtime_bootstrap()

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from langchain_core.messages import HumanMessage

from src.graph.builder import build_compiled_graph
from src.utils.config import LeadershipAgentConfig
from src.utils.observability import build_langfuse_callbacks


def _per_turn_state(user_query: str) -> dict:
    return {
        "question": user_query,
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)],
        "tool_call_count": 0,
        "rag_sources": [],
        "web_sources": [],
        "merged_context": "",
        "final_answer": "",
        "analysis_stdout": "",
        "analysis_stderr": "",
        "analysis_image_paths": [],
        "html_report_path": "",
        "is_safe": True,
        "error_message": None,
    }


def main() -> None:
    cfg = LeadershipAgentConfig.load(ROOT / "leadership_agent_config.json")
    graph = build_compiled_graph(cfg)
    thread_id = str(uuid4())

    print("Adobe Leadership AI Agent — ready")
    print(f"Session thread_id: {thread_id}")
    print("Commands: type a question, or 'exit' to quit.\n")

    while True:
        try:
            query = input("\n> Leadership question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in ("exit", "quit", ""):
            break

        config: dict = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(48, cfg.react_agent.max_tool_rounds * 4 + 24),
        }
        callbacks = build_langfuse_callbacks(cfg)
        if callbacks:
            config["callbacks"] = callbacks
        result = graph.invoke(_per_turn_state(query), config=config)

        if result.get("is_safe") is False:
            print(f"\nBlocked: {result.get('error_message', 'Input not allowed.')}\n")
            continue

        print("\n--- Answer ---\n")
        print(result.get("final_answer", ""))
        hr = result.get("html_report_path")
        if hr:
            print(f"\nOpen HTML report: {hr}")
        print("\n--------------")


if __name__ == "__main__":
    main()
