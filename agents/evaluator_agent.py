"""
Evaluator Agent — scores answers using LLM-as-judge.

Used by Tree of Thoughts for branch evaluation and by Reflexion
as the feedback signal for self-reflection.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from core.llm_interface import BaseLLM
from core.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """Scores an answer given a task using LLM-as-judge."""

    def __init__(self, llm: BaseLLM, prompt_builder: PromptBuilder) -> None:
        self.llm = llm
        self.pb = prompt_builder

    def evaluate(self, task: str, answer: str) -> float:
        """Score an answer on a scale of 0.0–1.0.

        Uses the LLM as an impartial judge. Falls back to 0.5 if
        the score cannot be parsed from the response.
        """
        prompt = self.pb.build("evaluate", task=task, answer=answer)

        response = self.llm.generate(
            prompt,
            max_tokens=16,
            temperature=0.1,  # low temp for consistent scoring
        )

        # Extract first decimal number from response
        match = re.search(r"(\d+\.?\d*)", response)
        if match:
            score = float(match.group(1))
            # Clamp to 0.0–1.0
            score = max(0.0, min(1.0, score))
            logger.info("Evaluator score: %.2f for task: %s", score, task[:60])
            return score

        logger.warning("Could not parse evaluator score from: %s", response)
        return 0.5
