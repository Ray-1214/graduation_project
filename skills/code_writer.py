"""
CodeWriter skill — delegates code generation to the CodingCoprocessor.

The MainAgent (Mistral, "thinking" model) decides *what* to write, then
invokes this skill via the ReAct loop.  The skill translates the natural-
language request into a ``CodingRequest`` and dispatches it to the
``CodingCoprocessor``, which hot-swaps to Qwen2.5-Coder for generation.

ReAct usage::

    Thought: I need to write a Fibonacci function.
    Action[code_writer]: Write a recursive function to calculate Fibonacci numbers
    Observation:
    ```python
    def fibonacci(n: int) -> int:
        ...
    ```
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from skills.registry import BaseSkill
from core.coding_coprocessor import CodingRequest, CodingTaskType

if TYPE_CHECKING:
    from core.coding_coprocessor import CodingCoprocessor

logger = logging.getLogger(__name__)


# ── Task-type keyword detection ──────────────────────────────────

_TASK_HINTS: list[tuple[re.Pattern, CodingTaskType]] = [
    (re.compile(r"\breview", re.I),         CodingTaskType.CODE_REVIEW),
    (re.compile(r"\btests?\b", re.I),       CodingTaskType.WRITE_TEST),
    (re.compile(r"\bdebug", re.I),          CodingTaskType.DEBUG),
    (re.compile(r"\bexplain", re.I),        CodingTaskType.EXPLAIN_CODE),
    (re.compile(r"\bedit\b|\bmodif", re.I), CodingTaskType.EDIT_CODE),
    (re.compile(r"\bskill", re.I),          CodingTaskType.WRITE_SKILL),
]


def _detect_task_type(text: str) -> CodingTaskType:
    """Infer the coding task type from natural-language keywords."""
    for pattern, task_type in _TASK_HINTS:
        if pattern.search(text):
            return task_type
    return CodingTaskType.WRITE_FUNCTION  # default


def _extract_context_code(text: str) -> tuple[str, str]:
    """Split *text* into (description, context_code).

    If the input contains a fenced code block, the code inside it is
    treated as ``context_code`` and the rest as ``description``.
    """
    match = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        desc = re.sub(r"```\w*\s*\n.*?```", "", text, flags=re.DOTALL).strip()
        return desc, code
    return text.strip(), ""


class CodeWriter(BaseSkill):
    """Generates code via the CodingCoprocessor (Qwen2.5-Coder).

    The skill parses the agent's natural-language request, infers the
    task type (write / edit / review / test / debug / explain / skill),
    and delegates to the coprocessor.

    Args:
        coprocessor: A :class:`CodingCoprocessor` instance.
    """

    def __init__(self, coprocessor: "CodingCoprocessor") -> None:
        self._coprocessor = coprocessor

    @property
    def name(self) -> str:
        return "code_writer"

    @property
    def description(self) -> str:
        return (
            "Generate, edit, review, test, or debug code. "
            "Input: a natural-language description of the coding task. "
            "Optionally include existing code in a ```python``` block for "
            "edits, reviews, or debugging. "
            "Output: the generated code with a brief explanation."
        )

    def execute(self, input_text: str) -> str:
        """Run the code-writing pipeline.

        Args:
            input_text: Natural-language instruction, optionally containing
                a fenced code block with context code.

        Returns:
            A formatted string with the generated code and explanation,
            or an error message if generation failed.
        """
        description, context_code = _extract_context_code(input_text)
        task_type = _detect_task_type(description)

        logger.info(
            "CodeWriter: task_type=%s, desc_len=%d, ctx_len=%d",
            task_type.value, len(description), len(context_code),
        )

        request = CodingRequest(
            task_type=task_type,
            description=description,
            context_code=context_code,
            language="python",
        )

        result = self._coprocessor.execute(request)

        if not result.success:
            return f"Code generation failed: {result.explanation}"

        # Format output for ReAct observation
        parts: list[str] = []

        if result.code:
            parts.append(f"```python\n{result.code}\n```")

        if result.explanation:
            parts.append(f"\n{result.explanation}")

        if result.warnings:
            parts.append("\n⚠️ Warnings:\n" + "\n".join(
                f"  - {w}" for w in result.warnings
            ))

        if result.swap_ms > 0:
            parts.append(f"\n(model swap: {result.swap_ms:.0f} ms)")

        return "\n".join(parts) if parts else "No output produced."

    def __repr__(self) -> str:
        return "CodeWriter(coprocessor=...)"
