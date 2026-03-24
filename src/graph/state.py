from __future__ import annotations

from operator import add
from typing import Annotated, List

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class AgentState(TypedDict, total=False):
    """ReAct + merged evidence + viz + HTML report path."""

    is_safe: NotRequired[bool]
    error_message: NotRequired[str]

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    user_query: NotRequired[str]

    tool_call_count: int
    rag_sources: Annotated[list[str], add]
    web_sources: Annotated[list[str], add]

    merged_context: str
    final_answer: str

    analysis_stdout: NotRequired[str]
    analysis_stderr: NotRequired[str]
    analysis_image_paths: NotRequired[List[str]]

    html_report_path: NotRequired[str]
