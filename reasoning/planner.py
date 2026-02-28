"""
Strategy Planner — selects the best reasoning strategy for a task.

Simple rule-based selector for MVP. Can be replaced with an LLM-based
meta-reasoner or a trained policy later.
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

StrategyName = Literal["cot", "tot", "react", "reflexion"]


# Keywords that suggest each strategy
_STRATEGY_HINTS = {
    "react": [
        "calculate", "compute", "search", "look up", "find",
        "what is", "how much", "file", "read", "write", "use tool",
    ],
    "tot": [
        "creative", "brainstorm", "multiple ways", "alternatives",
        "compare", "pros and cons", "complex", "plan",
    ],
    "reflexion": [
        "improve", "reflect", "learn from", "retry", "mistake",
    ],
}


class StrategyPlanner:
    """Selects a reasoning strategy based on task characteristics."""

    def select_strategy(self, task: str) -> StrategyName:
        """Analyze the task text and return the best strategy name.

        Falls back to 'cot' if no strong signal is detected.
        """
        task_lower = task.lower()

        # Score each strategy by keyword matches
        scores = {}
        for strategy, keywords in _STRATEGY_HINTS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[strategy] = score

        if scores:
            best = max(scores, key=scores.get)  # type: ignore[arg-type]
            logger.info(
                "StrategyPlanner: selected '%s' (scores: %s)",
                best, scores,
            )
            return best  # type: ignore[return-value]

        logger.info("StrategyPlanner: defaulting to 'cot'")
        return "cot"
