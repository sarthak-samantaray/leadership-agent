from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.config import LeadershipAgentConfig


def make_chat_llm(cfg: LeadershipAgentConfig) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        google_api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        timeout=cfg.llm.timeout_seconds,
        max_output_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
    )
