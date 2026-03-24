"""Structured outputs for prompt-based guardrails."""

from pydantic import BaseModel, Field


class PromptGuardrailVerdict(BaseModel):
    """LLM classifier: whether the user turn is appropriate for this assistant."""

    allowed: bool = Field(
        description=(
            "True if the message belongs in an executive business/leadership assistant "
            "(strategy, finance, markets, Adobe, risk, governance, follow-ups in that thread)."
        )
    )
    brief_reason: str = Field(
        default="",
        description="If allowed is false: one short, polite reason for the user. Empty if allowed.",
    )
