#!/usr/bin/env python3
"""Streamlit UI for the Adobe Leadership Agent — sidebar API keys, Q&A, HTML report preview."""

from __future__ import annotations

import base64
import mimetypes
import re
import sys
import webbrowser
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime_bootstrap import apply_runtime_bootstrap

apply_runtime_bootstrap()

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage

from src.graph.builder import build_compiled_graph
from src.utils.config import LeadershipAgentConfig
from src.utils.observability import build_langfuse_callbacks


def _load_base_config() -> LeadershipAgentConfig:
    return LeadershipAgentConfig.load(ROOT / "leadership_agent_config.json")


def _ensure_sidebar_defaults(cfg: LeadershipAgentConfig) -> None:
    """Seed session_state from JSON once so sidebar widgets have sensible defaults."""
    defaults: dict = {
        "sb_gemini": cfg.llm.api_key,
        "sb_model": cfg.llm.model,
        "sb_tavily": cfg.web_search.api_key,
        "sb_cohere": cfg.rag.cohere.api_key,
        "sb_lf_pub": cfg.langfuse.public_key,
        "sb_lf_sec": cfg.langfuse.secret_key,
        "sb_lf_url": cfg.langfuse.base_url,
        "sb_ws_enabled": cfg.web_search.enabled,
        "sb_cohere_enabled": cfg.rag.cohere.enabled,
        "sb_lf_enabled": cfg.langfuse.enabled,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _merge_config_from_sidebar(base: LeadershipAgentConfig) -> LeadershipAgentConfig:
    """Apply sidebar values; empty fields keep JSON / env defaults."""
    gemini = (st.session_state.get("sb_gemini") or "").strip() or base.llm.api_key
    model = (st.session_state.get("sb_model") or "").strip() or base.llm.model
    tavily = (st.session_state.get("sb_tavily") or "").strip() or base.web_search.api_key
    cohere_k = (st.session_state.get("sb_cohere") or "").strip() or base.rag.cohere.api_key
    lf_pub = (st.session_state.get("sb_lf_pub") or "").strip() or base.langfuse.public_key
    lf_sec = (st.session_state.get("sb_lf_sec") or "").strip() or base.langfuse.secret_key
    lf_url = (st.session_state.get("sb_lf_url") or "").strip() or base.langfuse.base_url

    ws_on = bool(st.session_state.get("sb_ws_enabled", base.web_search.enabled))
    co_on = bool(st.session_state.get("sb_cohere_enabled", base.rag.cohere.enabled))
    lf_on = bool(st.session_state.get("sb_lf_enabled", base.langfuse.enabled))

    new_llm = base.llm.model_copy(update={"api_key": gemini, "model": model})
    new_ws = base.web_search.model_copy(update={"enabled": ws_on, "api_key": tavily})
    new_cohere = base.rag.cohere.model_copy(update={"enabled": co_on, "api_key": cohere_k})
    new_rag = base.rag.model_copy(update={"cohere": new_cohere})
    new_lf = base.langfuse.model_copy(
        update={
            "enabled": lf_on,
            "public_key": lf_pub,
            "secret_key": lf_sec,
            "base_url": lf_url.rstrip("/") or base.langfuse.base_url,
        }
    )
    return base.model_copy(
        update={
            "llm": new_llm,
            "web_search": new_ws,
            "rag": new_rag,
            "langfuse": new_lf,
        }
    )


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


def _html_with_inline_images(html: str, html_file: Path) -> str:
    """Embed local images as data: URLs so Streamlit's iframe preview can show them.

    Browsers block file:// assets inside http:// iframes; base64 data URLs are same-document.
    """
    base = html_file.parent.resolve()

    def fix_src(m: re.Match[str]) -> str:
        src = (m.group(1) or "").strip()
        if not src or src.startswith(("data:", "http://", "https://", "file:")):
            return m.group(0)
        path = Path(src)
        if not path.is_absolute():
            path = (base / src).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            return m.group(0)
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/png"
        try:
            b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return m.group(0)
        return f'src="data:{mime};base64,{b64}"'

    return re.sub(r'src="([^"]+)"', fix_src, html)


def _inject_css() -> None:
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=Instrument+Serif&display=swap');
  html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
  .main-header {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    color: #12141a;
    margin-bottom: 0.15rem;
  }
  .subtle {
    color: #5c6370;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }
  div[data-testid="stSidebar"] { border-right: 1px solid #e8eaef; }
</style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Adobe Leadership Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    base_cfg = _load_base_config()
    _ensure_sidebar_defaults(base_cfg)

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.caption(
            "Keys stay in this browser session only (not saved to disk). "
            "Leave blank to use values from `leadership_agent_config.json` / `.env`."
        )

        st.text_input(
            "Gemini API key",
            type="password",
            key="sb_gemini",
            help="Google AI Studio API key",
        )
        st.text_input(
            "Gemini model",
            key="sb_model",
            help="e.g. gemini-2.5-flash-lite",
        )

        st.divider()
        st.markdown("**Web search (Tavily)**")
        st.toggle("Enable web search", key="sb_ws_enabled")
        st.text_input(
            "Tavily API key",
            type="password",
            key="sb_tavily",
        )

        st.divider()
        st.markdown("**Cohere rerank**")
        st.toggle("Enable Cohere rerank", key="sb_cohere_enabled")
        st.text_input(
            "Cohere API key",
            type="password",
            key="sb_cohere",
        )

        st.divider()
        st.markdown("**Langfuse (optional)**")
        st.toggle("Enable Langfuse tracing", key="sb_lf_enabled")
        st.text_input("Langfuse public key", type="password", key="sb_lf_pub")
        st.text_input("Langfuse secret key", type="password", key="sb_lf_sec")
        st.text_input("Langfuse base URL", key="sb_lf_url")

        if st.button("🔄 New conversation thread"):
            st.session_state.thread_id = str(uuid4())
            st.session_state.pop("last_agent_result", None)
            st.rerun()

    cfg = _merge_config_from_sidebar(base_cfg)

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())

    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.markdown('<p class="main-header">Leadership insight assistant</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="subtle">Grounded answers from your PDFs + web, with charts and an HTML brief.</p>',
            unsafe_allow_html=True,
        )

        q = st.text_area(
            "Your question",
            height=120,
            placeholder="e.g. Summarize Adobe’s key risks and revenue trends from filings…",
            key="user_question",
        )
        run = st.button("Run agent", type="primary", use_container_width=True)

    with col2:
        st.info(
            "**Tip:** Run `python ingest.py` after adding PDFs to `data/documents/`. "
            f"Thread id: `{st.session_state.thread_id[:8]}…`"
        )

    if run and (not q or not str(q).strip()):
        st.warning("Enter a question first.")
        run = False

    if run and q and str(q).strip():
        graph = build_compiled_graph(cfg)
        config: dict = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": max(48, cfg.react_agent.max_tool_rounds * 4 + 24),
        }
        callbacks = build_langfuse_callbacks(cfg)
        if callbacks:
            config["callbacks"] = callbacks

        with st.spinner("Running LangGraph (guardrails → ReAct → analysis → HTML)…"):
            result = graph.invoke(_per_turn_state(q.strip()), config=config)
        st.session_state["last_agent_result"] = result

    result = st.session_state.get("last_agent_result")
    if not result:
        return

    if result.get("is_safe") is False:
        st.error(result.get("error_message", "Blocked by guardrails."))
        return

    answer = result.get("final_answer") or ""
    st.markdown("### Answer")
    st.markdown(answer)

    hr = result.get("html_report_path")
    if hr:
        p = Path(hr)
        st.markdown("### Report")
        c1, c2 = st.columns([1, 1])
        with c1:
            if p.is_file():
                st.download_button(
                    label="⬇️ Download HTML report",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="text/html",
                    use_container_width=True,
                    key="dl_html_report",
                )
        with c2:
            if p.is_file():
                if st.button(
                    "🌐 Open report in default browser",
                    key="open_report_browser",
                    use_container_width=True,
                    help="Opens the HTML file with your OS default browser (local Streamlit only).",
                ):
                    try:
                        webbrowser.open(p.resolve().as_uri())
                    except Exception as ex:
                        st.warning(f"Could not open browser: {ex}")
        st.caption(
            f"Path: `{p}` — Browsers block `file://` links from this page; use the button or download."
        )

        if p.is_file():
            try:
                raw = p.read_text(encoding="utf-8")
                fixed = _html_with_inline_images(raw, p)
                st.markdown("**Preview** (images embedded for display)")
                components.html(fixed, height=720, scrolling=True)
            except Exception as e:
                st.caption(f"Preview unavailable: {e}")


if __name__ == "__main__":
    main()
