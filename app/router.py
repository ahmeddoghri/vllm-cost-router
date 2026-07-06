"""Complexity-based model routing: cheap requests go to the small tier, only
genuinely hard ones pay for the large model. This is the actual cost lever --
not every request needs the expensive model, and blindly routing everything
to it is the single biggest source of wasted inference spend.
"""
from __future__ import annotations

import re

_CODE_MARKERS = re.compile(r"```|def |class |SELECT |import |function\(")
_REASONING_MARKERS = re.compile(
    r"\b(why|explain|analyze|compare|prove|derive|design|architecture|trade-?offs?)\b",
    re.IGNORECASE,
)


def complexity_score(prompt: str, max_tokens: int = 256) -> float:
    """Heuristic complexity in [0, 1]. Longer prompts, code, multi-step
    reasoning language, and larger requested outputs all push toward the
    large tier; short factual asks stay on the cheap one.
    """
    length_component = min(1.0, len(prompt) / 800)
    tokens_component = min(1.0, max_tokens / 1024)
    # a genuine complexity signal (code, or explicit multi-step reasoning
    # language) should be enough on its own to cross the routing threshold,
    # even for an otherwise short prompt
    code_component = 0.45 if _CODE_MARKERS.search(prompt) else 0.0
    reasoning_component = 0.45 if _REASONING_MARKERS.search(prompt) else 0.0
    score = 0.3 * length_component + 0.1 * tokens_component + code_component + reasoning_component
    return max(0.0, min(1.0, score))


def choose_tier(prompt: str, max_tokens: int, threshold: float,
                small_model: str, large_model: str) -> tuple[str, float]:
    score = complexity_score(prompt, max_tokens)
    tier = large_model if score >= threshold else small_model
    return tier, score
