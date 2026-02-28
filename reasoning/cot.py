"""
Chain of Thought — single-pass reasoning baseline.

The simplest strategy: append "think step by step" to the prompt
and return the LLM's response. Serves as a baseline for comparison
with multi-step strategies.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.llm_interface import BaseLLM
from core.prompt_builder import PromptBuilder
from memory.episodic_log import EpisodicLog

logger = logging.getLogger(__name__)


class ChainOfThought:
    """Single-pass Chain of Thought reasoning."""

    def __init__(self, llm: BaseLLM, prompt_builder: PromptBuilder) -> None:
        self.llm = llm
        self.pb = prompt_builder

    def run(self, task: str, episode: Optional[EpisodicLog] = None) -> str:
        """Generate a CoT response for the given task."""
        prompt = self.pb.build("cot", task=task)

        if episode:
            episode.log_step("thought", f"Using Chain of Thought for: {task}")

        logger.info("CoT: generating response …")
        response = self.llm.generate(prompt)

        if episode:
            episode.log_step("finish", response)

        return response
