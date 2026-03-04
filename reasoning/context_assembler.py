"""
ContextAssembler — builds the full prompt context for CompoundReasoner.

Concatenates skill blocks, reflexion lessons, RAG context, and the
task itself into a single effective prompt string, in a standard order.
"""

from __future__ import annotations

from typing import Optional


class ContextAssembler:
    """Assembles prompt context from multiple sources.

    Ordering:
      1. Skill block (from SkillRetriever)
      2. Reflexion lessons
      3. RAG context
      4. Structured thinking plan (Phase ①)
      5. Task description
    """

    def assemble(
        self,
        task: str,
        skills_block: Optional[str] = None,
        lessons: Optional[str] = None,
        rag_context: Optional[str] = None,
        thinking_plan: Optional[str] = None,
    ) -> str:
        """Combine all context sources into one prompt string.

        Args:
            task:           Original task description.
            skills_block:   Formatted skill retrieval block.
            lessons:        Past reflexion lessons.
            rag_context:    Retrieved RAG context.
            thinking_plan:  Phase ① structured thinking plan.

        Returns:
            Combined prompt string.
        """
        parts: list[str] = []

        if skills_block:
            parts.append(skills_block)

        if lessons:
            parts.append(lessons)

        if rag_context:
            parts.append(rag_context)

        if thinking_plan:
            parts.append(
                f"[任務分析計劃]\n{thinking_plan}\n[計劃結束]"
            )

        parts.append(task)
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return "ContextAssembler()"
