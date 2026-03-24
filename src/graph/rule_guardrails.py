"""Lightweight rule-based input checks: block obvious non-business / casual / injection prompts."""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Prompt-injection / jailbreak-ish phrases (substring match, case-insensitive).
_INJECTION = (
    "ignore previous",
    "ignore all prior",
    "disregard the above",
    "system prompt",
    "you are now",
    "developer mode",
    "jailbreak",
    "dan mode",
    "simulate a",
    "bypass safety",
    "reveal your instructions",
)

# Clear non-business / “random” intents (blocked when the message is short enough to be only that).
_CASUAL_JUNK = (
    "tell me a joke",
    "write me a joke",
    "write a poem",
    "sing me",
    "what's the weather",
    "weather in",
    "who won the",
    "recipe for",
    "how do i cook",
    "movie plot",
    "video game cheat",
    "write python code to hack",
    "translate this to",
)

# Very short greetings / noise with no business angle.
_TRIVIAL_ONLY = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|thanks|thank you|ok|okay|k|bye|lol|haha|test)\s*!*\s*$",
    re.IGNORECASE,
)

# Broad business / leadership / finance / company context (if any match → allow unless injection).
_BUSINESS_SIGNAL = re.compile(
    r"\b("
    r"adobe|business|leadership|executive|ceo|cfo|cto|board|strategy|revenue|earnings|"
    r"margin|profit|growth|forecast|guidance|quarter|annual|fiscal|year|yoy|financial|"
    r"market|competitor|customer|segment|product|saas|subscription|cloud|digital|"
    r"acquisition|merger|risk|compliance|sec|filing|10-?k|10-?q|8-?k|shareholder|"
    r"stock|nasdaq|eps|ebitda|cash flow|operating|sales|kpi|benchmark|industry|"
    r"macro|economic|regulation|workforce|talent|culture|transformation|innovation|"
    r"report|analysis|trend|performance|metric|roi|pipeline"
    r")\b",
    re.IGNORECASE,
)


def evaluate_input(text: str) -> Tuple[bool, Optional[str]]:
    """
    Returns (allowed, error_message).

    Designed to be **permissive**: only block obvious cases—injection patterns, trivial
    one-word greetings, and clear casual-junk phrases. Finer filtering is left to the
    prompt-based guardrail.
    """
    q = (text or "").strip()
    if len(q) < 4:
        return False, "Please enter a question (at least a few characters)."

    low = q.lower()
    if any(p in low for p in _INJECTION):
        return False, "Blocked: disallowed prompt pattern."

    if any(p in low for p in _CASUAL_JUNK) and len(q) < 160:
        return False, "That looks off-topic. Ask a business or leadership question (e.g. strategy, financials, markets)."

    if _TRIVIAL_ONLY.match(q):
        return False, "Ask a concrete business or leadership question—not just a greeting."

    if _BUSINESS_SIGNAL.search(q):
        return True, None

    if re.search(r"[\d%$]", q):
        return True, None

    if len(q) >= 80:
        return True, None

    # No strong keyword/number/length signal — defer to prompt guardrail (do not block here).
    return True, None
