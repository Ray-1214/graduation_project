"""
Compound reasoning result types.

Dataclasses produced and consumed by :class:`CompoundReasoner` and its
sub-components (HallucinationGuard, ContextAssembler).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ThinkingResult:
    """Output of Phase ① Structured Thinking.

    Attributes:
        task_type:              E.g. "事實查詢", "推理計算", "多步驟操作", "創意生成".
        complexity:             "simple" | "moderate" | "complex".
        tools_needed:           List of tool names the plan requires.
        recommended_strategy:   "cot_only" | "react_cot" | "react_tot" | "full_compound".
        sub_goals:              Ordered list of sub-goal descriptions.
        raw_analysis:           Full LLM analysis text (for logging).
    """

    task_type: str = "unknown"
    complexity: str = "moderate"
    tools_needed: List[str] = field(default_factory=list)
    recommended_strategy: str = "cot_only"
    sub_goals: List[str] = field(default_factory=list)
    raw_analysis: str = ""


@dataclass
class SelfCheckResult:
    """Output of a Grounded Self-Check step.

    Attributes:
        passed:       True if no critical issues found.
        needs_replan: True if the remaining plan should be revised.
        issues:       List of detected inconsistencies / concerns.
        progress_pct: Estimated progress toward final answer (0–100).
    """

    passed: bool = True
    needs_replan: bool = False
    issues: List[str] = field(default_factory=list)
    progress_pct: float = 0.0


@dataclass
class VerificationResult:
    """Output of Phase ③ Answer Verification Gate.

    Attributes:
        verified:       True if the answer passed verification.
        confidence:     Overall confidence score (0.0–1.0).
        flagged_claims: Factual claims that could not be verified.
        report:         Human-readable verification summary.
    """

    verified: bool = True
    confidence: float = 1.0
    flagged_claims: List[str] = field(default_factory=list)
    report: str = ""


@dataclass
class CompoundResult:
    """Final output of :meth:`CompoundReasoner.run`.

    Attributes:
        answer:           The final answer string.
        confidence:       Overall confidence (0.0–1.0).
        strategy_used:    Which execution mode was used.
        thinking:         Phase ① plan (None if force_strategy was set).
        verification:     Phase ③ verification report.
        steps_taken:      Number of reasoning steps executed.
        think_steps:      Number of think-tool invocations.
        self_checks:      Number of grounded self-check invocations.
    """

    answer: str = ""
    confidence: float = 1.0
    strategy_used: str = "cot_only"
    thinking: Optional[ThinkingResult] = None
    verification: Optional[VerificationResult] = None
    steps_taken: int = 0
    think_steps: int = 0
    self_checks: int = 0
