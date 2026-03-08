"""
Coding Coprocessor — structured code generation via a dedicated coding model.

The MainAgent (CPU) sends a CodingRequest to this coprocessor, which uses
ModelManager to hot-swap to the coding model, generate code, and return a
CodingResult.  ModelManager handles VRAM scheduling transparently.

Usage::

    from core.llm_interface import ModelManager
    from core.coding_coprocessor import CodingCoprocessor, CodingRequest, CodingTaskType

    mgr = ModelManager(config)
    coproc = CodingCoprocessor(mgr)
    result = coproc.execute(CodingRequest(
        task_type=CodingTaskType.WRITE_FUNCTION,
        description="Write a binary search function",
        language="python",
    ))
    print(result.code)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from core.llm_interface import ModelManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Enums & Data Models
# ═══════════════════════════════════════════════════════════════════

class CodingTaskType(Enum):
    """Types of coding tasks the coprocessor can handle."""

    WRITE_FUNCTION = "write_function"      # Write a new function from scratch
    EDIT_CODE      = "edit_code"           # Modify existing code
    CODE_REVIEW    = "code_review"         # Review code and find bugs
    WRITE_TEST     = "write_test"          # Write unit tests
    EXPLAIN_CODE   = "explain_code"        # Explain what code does
    DEBUG          = "debug"               # Find and fix bugs
    WRITE_SKILL    = "write_skill"         # Write a new BaseSkill subclass


@dataclass
class CodingRequest:
    """Structured request sent from the MainAgent to the CodingCoprocessor.

    Attributes:
        task_type: The type of coding task to perform.
        description: Natural-language description of what to produce.
        context_code: Existing source code for context (edit, review, etc.).
        constraints: Additional constraints or requirements.
        target_file: Intended file path for the output code.
        language: Programming language for the output.
    """

    task_type: CodingTaskType
    description: str
    context_code: str = ""
    constraints: List[str] = field(default_factory=list)
    target_file: str = ""
    language: str = "python"


@dataclass
class CodingResult:
    """Structured result returned by the CodingCoprocessor.

    Attributes:
        success: Whether code generation completed without error.
        code: The generated source code (extracted from code blocks).
        explanation: Any explanatory text the model produced.
        file_path: Suggested file path for the generated code.
        warnings: Non-fatal issues noted during generation.
        swap_ms: Time spent on model swap (0 if already loaded).
    """

    success: bool
    code: str = ""
    explanation: str = ""
    file_path: str = ""
    warnings: List[str] = field(default_factory=list)
    swap_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════
#  Prompt Templates (delegated to core.coding_prompts)
# ═══════════════════════════════════════════════════════════════════

_GENERIC_FALLBACK = (
    "You are an expert {language} programmer.\n"
    "{description}\n\n"
    "{context_code}"
    "{constraints}"
    "Respond with a ```{language}``` code block.\n"
)


# ═══════════════════════════════════════════════════════════════════
#  CodingCoprocessor
# ═══════════════════════════════════════════════════════════════════

class CodingCoprocessor:
    """Structured code-generation coprocessor.

    Receives :class:`CodingRequest` objects, builds a prompt from the
    appropriate template, delegates generation to :class:`ModelManager`
    (which hot-swaps to the coding model), and parses the output into a
    :class:`CodingResult`.

    Args:
        model_manager: Shared :class:`ModelManager` instance.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, request: CodingRequest) -> CodingResult:
        """Execute a coding task and return a structured result.

        Args:
            request: The coding request to process.

        Returns:
            A :class:`CodingResult` with the generated code and metadata.
        """
        prompt = self._build_prompt(request)

        logger.info(
            "CodingCoprocessor: executing %s (lang=%s, target=%s)",
            request.task_type.value,
            request.language,
            request.target_file or "(none)",
        )

        # Record swap overhead
        swap_before = self._mm.stats["total_swap_ms"]

        try:
            raw_output = self._mm.generate(prompt, role="coding")
        except Exception as exc:
            logger.error("Code generation failed: %s", exc)
            return CodingResult(
                success=False,
                explanation=f"Generation error: {exc}",
                file_path=request.target_file,
            )

        swap_after = self._mm.stats["total_swap_ms"]
        swap_ms = swap_after - swap_before

        # Parse code block and explanation
        code, explanation = self._parse_output(raw_output, request.language)

        # Warnings
        warnings: List[str] = []
        if not code and request.task_type != CodingTaskType.EXPLAIN_CODE:
            warnings.append("No code block found in model output.")
        if len(code) > 10_000:
            warnings.append("Generated code exceeds 10 000 characters.")

        result = CodingResult(
            success=True,
            code=code,
            explanation=explanation or raw_output if not code else explanation,
            file_path=request.target_file,
            warnings=warnings,
            swap_ms=swap_ms,
        )

        logger.info(
            "CodingCoprocessor: done (%d chars code, %.0f ms swap, %d warnings)",
            len(code), swap_ms, len(warnings),
        )
        return result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, req: CodingRequest) -> str:
        """Build the full prompt from a template and request fields.

        Prefers the rich templates in :mod:`core.coding_prompts`.  Falls
        back to a generic template if the module is unavailable.

        Args:
            req: The coding request.

        Returns:
            The formatted prompt string.
        """
        # Try dedicated coding_prompts module first
        try:
            from core.coding_prompts import get_coding_prompt
            template = get_coding_prompt(req.task_type)
        except (ImportError, KeyError):
            template = _GENERIC_FALLBACK

        # Build optional blocks
        constraints = ""
        if req.constraints:
            items = "\n".join(f"- {c}" for c in req.constraints)
            constraints = f"### Constraints\n{items}\n\n"

        context_code = ""
        if req.context_code:
            context_code = req.context_code

        return template.format(
            language=req.language,
            description=req.description,
            context_code=context_code,
            constraints=constraints,
            target_file=req.target_file,
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(raw: str, language: str) -> Tuple[str, str]:
        """Extract code and explanation from the model's raw output.

        Looks for fenced code blocks matching *language* (or generic
        triple-backtick blocks) and separates them from prose.

        Args:
            raw: The raw text output from the model.
            language: Expected code language for block detection.

        Returns:
            A ``(code, explanation)`` tuple.
        """
        # Try language-specific fence first, then generic
        patterns = [
            rf"```{re.escape(language)}\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]

        code = ""
        for pattern in patterns:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                code = match.group(1).strip()
                break

        # Explanation is everything outside the code block
        if code:
            explanation = re.sub(
                r"```(?:\w+)?\s*\n.*?```", "", raw, flags=re.DOTALL
            ).strip()
        else:
            explanation = raw.strip()

        return code, explanation

    def __repr__(self) -> str:
        return f"CodingCoprocessor(manager={self._mm!r})"
