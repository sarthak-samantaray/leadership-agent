"""Normalize LangChain chat message content to plain text (Gemini uses list-of-blocks, not only str)."""

from __future__ import annotations

from typing import Any, List

# Gemini / multimodal: skip non-text parts so reasoning/thinking is not merged into "code" text.
_SKIP_MESSAGE_BLOCK_TYPES = frozenset(
    {"thinking", "thought", "function_call", "function_call_result", "inline_data"}
)


def message_content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = str(block.get("type") or "").lower()
                if btype in _SKIP_MESSAGE_BLOCK_TYPES:
                    continue
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            else:
                t = getattr(block, "text", None) or getattr(block, "content", None)
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts).strip()
    return str(content).strip()
