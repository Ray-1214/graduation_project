"""
Reflexion — post-task self-reflection and memory update.

After a reasoning episode completes, this module prompts the LLM
to reflect on its performance, extract lessons, and store them in
long-term memory for retrieval in future episodes.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from core.llm_interface import BaseLLM
from core.prompt_builder import PromptBuilder
from memory.episodic_log import EpisodicLog
from memory.long_term import LongTermMemory, ReflectionEntry

logger = logging.getLogger(__name__)


class Reflexion:
    """Post-task self-reflection with persistent memory update."""

    def __init__(
        self,
        llm: BaseLLM,
        prompt_builder: PromptBuilder,
        long_term_memory: LongTermMemory,
    ) -> None:
        self.llm = llm
        self.pb = prompt_builder
        self.ltm = long_term_memory

    def reflect(
        self,
        task: str,
        episode: EpisodicLog,
        outcome: str,
        score: float,
    ) -> ReflectionEntry:
        """Generate a self-reflection and store it in long-term memory.

        Args:
            task: The original task description.
            episode: The full episodic trace of the reasoning episode.
            outcome: The final answer/result produced.
            score: Evaluation score (0.0–1.0) from the evaluator.

        Returns:
            The stored ReflectionEntry.
        """
        trajectory = episode.trajectory_str()

        prompt = self.pb.build(
            "reflexion",
            task=task,
            trajectory=trajectory,
            outcome=outcome,
            score=f"{score:.2f}",
        )

        logger.info("Reflexion: generating self-reflection …")
        reflection_text = self.llm.generate(prompt, max_tokens=512)

        # Extract lessons (lines starting with numbers or bullets)
        lessons = self._extract_lessons(reflection_text)

        entry = ReflectionEntry(
            task=task,
            reflection=reflection_text,
            lessons=lessons,
            score=score,
        )

        self.ltm.store(entry)
        logger.info(
            "Reflexion stored %d lessons (score=%.2f).",
            len(lessons), score,
        )

        return entry

    def get_relevant_lessons(
        self,
        task: str,
        top_k: int = 3,
    ) -> str:
        """Retrieve past lessons relevant to a new task."""
        keywords = task.lower().split()[:10]
        entries = self.ltm.retrieve(keywords, top_k=top_k)
        if not entries:
            return ""

        parts = ["Past lessons for similar tasks:"]
        for i, entry in enumerate(entries, 1):
            lessons_str = "; ".join(entry.lessons) if entry.lessons else entry.reflection[:200]
            parts.append(f"{i}. (score={entry.score:.2f}) {lessons_str}")
        return "\n".join(parts)

    @staticmethod
    def _extract_lessons(text: str) -> List[str]:
        """Parse numbered or bulleted items from reflection text."""
        # Match lines like "1. ...", "- ...", "• ..."
        lines = re.findall(r"(?:^\d+\.\s*|^[-•]\s*)(.+)", text, re.MULTILINE)
        if lines:
            return [line.strip() for line in lines if line.strip()]
        # Fallback: return the whole text as a single lesson
        return [text.strip()] if text.strip() else []
