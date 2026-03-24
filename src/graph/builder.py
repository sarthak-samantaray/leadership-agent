"""LangGraph: agent ↔ tools → finalize → analysis (charts) → HTML report."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.graph.analysis_runner import run_python_viz_analysis
from src.graph.html_report import write_leadership_html_report
from src.graph.react_tools import make_react_tools
from src.graph.rule_guardrails import evaluate_input
from src.graph.schemas import PromptGuardrailVerdict
from src.graph.state import AgentState
from src.utils.config import LeadershipAgentConfig
from src.utils.llm_factory import make_chat_llm
from src.utils.logger import get_logger
from src.utils.message_content import message_content_to_str
from src.utils.prompts import LEADERSHIP_AGENT_SYSTEM, PROMPT_GUARDRAIL_SYSTEM

_log = get_logger(__name__)


def _route_after_rule(state: AgentState) -> Literal["prompt_guardrail", "stop"]:
    if state.get("is_safe") is False:
        return "stop"
    return "prompt_guardrail"


def _route_after_prompt(state: AgentState) -> Literal["agent", "stop"]:
    if state.get("is_safe") is False:
        return "stop"
    return "agent"


def _should_continue(
    state: AgentState, max_tool_rounds: int
) -> Literal["tools", "finalize"]:
    msgs = state.get("messages") or []
    if not msgs:
        return "finalize"
    last = msgs[-1]
    tcc = int(state.get("tool_call_count") or 0)
    if isinstance(last, AIMessage) and (last.tool_calls or []) and tcc < max_tool_rounds:
        return "tools"
    return "finalize"


def build_compiled_graph(cfg: LeadershipAgentConfig):
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    llm = make_chat_llm(cfg)
    tools = make_react_tools(cfg, project_root)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    llm_guard = llm.with_structured_output(PromptGuardrailVerdict)
    max_rounds = cfg.react_agent.max_tool_rounds
    timeout_viz = max(30, min(cfg.llm.timeout_seconds, 120))

    def rule_guardrail_node(state: AgentState) -> AgentState:
        q = (state.get("question") or state.get("user_query") or "").strip()
        ok, err = evaluate_input(q)
        if ok:
            return {"is_safe": True, "error_message": None}
        msg = err or "Input not allowed."
        return {
            "is_safe": False,
            "error_message": msg,
            "final_answer": msg,
        }

    def prompt_guardrail_node(state: AgentState) -> AgentState:
        q = (state.get("question") or state.get("user_query") or "").strip()
        try:
            verdict: PromptGuardrailVerdict = llm_guard.invoke(
                [
                    SystemMessage(content=PROMPT_GUARDRAIL_SYSTEM),
                    HumanMessage(content=q),
                ]
            )
            if verdict.allowed:
                return {"is_safe": True, "error_message": None}
            reason = (verdict.brief_reason or "").strip() or (
                "This assistant only handles business and leadership questions."
            )
            return {
                "is_safe": False,
                "error_message": reason,
                "final_answer": reason,
            }
        except Exception:
            _log.exception("Prompt guardrail LLM call failed")
            msg = "Could not validate your query (guardrail error). Try again."
            return {
                "is_safe": False,
                "error_message": msg,
                "final_answer": msg,
            }

    def agent_node(state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        messages = list(state.get("messages") or [])
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=LEADERSHIP_AGENT_SYSTEM)] + messages
        response = llm_with_tools.invoke(messages, ensure_config(config))
        return {"messages": [response]}

    def tools_node(state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        last = (state.get("messages") or [])[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"tool_call_count": int(state.get("tool_call_count") or 0)}
        rag_new: list[str] = []
        web_new: list[str] = []
        tool_messages: list[ToolMessage] = []
        for tc in last.tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
            tid = tid or "tool_call"
            print(f"  [tool] {name}({args})", flush=True)
            fn = tools_by_name.get(name)
            if not fn:
                text = f"Unknown tool: {name}"
            else:
                text = fn.invoke(args)
            if name == "rag_search":
                rag_new.extend(re.findall(r"\[SOURCE \d+: ([^\]]+)\]", text))
            elif name == "web_search":
                web_new.extend(re.findall(r"\[WEB SOURCE \d+: ([^\]]+)\]", text))
            tool_messages.append(ToolMessage(content=text, tool_call_id=tid))
        prev = int(state.get("tool_call_count") or 0)
        return {
            "messages": tool_messages,
            "tool_call_count": prev + 1,
            "rag_sources": list(dict.fromkeys(rag_new)),
            "web_sources": list(dict.fromkeys(web_new)),
        }

    def finalize_node(state: AgentState) -> AgentState:
        msgs = list(state.get("messages") or [])
        tool_texts = [
            message_content_to_str(m.content) for m in msgs if isinstance(m, ToolMessage)
        ]
        merged = "\n\n=== TOOL OUTPUT ===\n\n".join(tool_texts) if tool_texts else ""
        final = ""
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                txt = message_content_to_str(msg.content)
                if txt and not (msg.tool_calls or []):
                    final = txt
                    break
        if not final:
            for msg in reversed(msgs):
                if isinstance(msg, AIMessage):
                    txt = message_content_to_str(msg.content)
                    if txt:
                        final = txt
                        break
        return {
            "final_answer": final or "(No final response produced.)",
            "merged_context": merged,
        }

    def analysis_agent_node(state: AgentState) -> AgentState:
        q = (state.get("question") or state.get("user_query") or "").strip()
        evidence = (state.get("merged_context") or "").strip()
        tail = (state.get("final_answer") or "").strip()
        ev = (evidence + "\n\n" + tail) if tail else evidence
        out, err, paths = run_python_viz_analysis(
            llm, q, ev, reports_dir, timeout_seconds=timeout_viz
        )
        return {
            "analysis_stdout": out,
            "analysis_stderr": err or "",
            "analysis_image_paths": paths,
        }

    def html_report_node(state: AgentState) -> AgentState:
        path = write_leadership_html_report(
            question=state.get("question") or "",
            final_answer=state.get("final_answer") or "",
            merged_context=state.get("merged_context") or "",
            analysis_stdout=state.get("analysis_stdout") or "",
            analysis_stderr=state.get("analysis_stderr") or "",
            image_paths=list(state.get("analysis_image_paths") or []),
            cfg=cfg,
            project_root=project_root,
        )
        fa = (state.get("final_answer") or "").strip()
        suffix = f"\n\n**HTML report:** `{path}`"
        return {
            "html_report_path": str(path),
            "final_answer": (fa + suffix).strip(),
        }

    graph = StateGraph(AgentState)
    graph.add_node("rule_guardrail", rule_guardrail_node)
    graph.add_node("prompt_guardrail", prompt_guardrail_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("analysis_agent", analysis_agent_node)
    graph.add_node("html_report", html_report_node)
    graph.set_entry_point("rule_guardrail")
    graph.add_conditional_edges(
        "rule_guardrail",
        _route_after_rule,
        {"prompt_guardrail": "prompt_guardrail", "stop": END},
    )
    graph.add_conditional_edges(
        "prompt_guardrail",
        _route_after_prompt,
        {"agent": "agent", "stop": END},
    )
    graph.add_conditional_edges(
        "agent",
        lambda s: _should_continue(s, max_rounds),
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", "analysis_agent")
    graph.add_edge("analysis_agent", "html_report")
    graph.add_edge("html_report", END)

    mem_path = cfg.resolve(cfg.paths.memory_db)
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(mem_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)