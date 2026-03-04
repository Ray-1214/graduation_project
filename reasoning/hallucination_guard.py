"""
HallucinationGuard — Answer Verification Gate (Phase ③).

Extracts factual claims from the answer, cross-checks them against
the KnowledgeStore and recent tool observations, and flags any
unverifiable assertions.

References
~~~~~~~~~~
- Huang et al. *"LLMs Cannot Self-Correct Reasoning Yet"*
  (ICLR 2024, arXiv:2310.01798) — motivates grounding on external
  signals rather than pure self-assessment.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List, Optional

from reasoning.compound_result import VerificationResult

if TYPE_CHECKING:
    from core.llm_interface import BaseLLM
    from memory.episodic_log import EpisodicLog
    from rag.knowledge_store import KnowledgeStore
    from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class HallucinationGuard:
    """Verifies factual claims in an answer before returning it.

    Args:
        llm:              LLM backend for claim extraction.
        knowledge_store:  KnowledgeStore for cross-validation.
        skill_registry:   Optional tool registry for live verification.
        confidence_threshold:  Claims below this similarity are flagged.
    """

    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,
        knowledge_store: Optional["KnowledgeStore"] = None,
        skill_registry: Optional["SkillRegistry"] = None,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.llm = llm
        self.knowledge_store = knowledge_store
        self.skill_registry = skill_registry
        self.confidence_threshold = confidence_threshold

    def verify(
        self,
        answer: str,
        task: str,
        trace_steps: List[str],
        episode: Optional["EpisodicLog"] = None,
    ) -> VerificationResult:
        """Run the verification gate on an answer.

        Steps:
          1. Extract factual claims from the answer.
          2. Cross-check each claim against KnowledgeStore.
          3. For unverified claims, attempt tool verification.
          4. Return flagged claims and overall confidence.

        Args:
            answer:      The candidate answer text.
            task:        Original task description.
            trace_steps: List of observation strings from execution.
            episode:     Optional episodic log for recording.

        Returns:
            :class:`VerificationResult` with confidence and flags.
        """
        # Step 1: Extract claims
        claims = self._extract_claims(answer)
        if not claims:
            if episode:
                episode.log_step("self_check", "No factual claims to verify")
            return VerificationResult(
                verified=True, confidence=0.9,
                report="No factual claims extracted; answer accepted.",
            )

        # Step 2: Cross-check against knowledge store
        flagged: List[str] = []
        verified_count = 0

        for claim in claims:
            is_verified = self._check_against_knowledge(claim)
            if not is_verified:
                is_verified = self._check_against_trace(claim, trace_steps)
            if not is_verified:
                is_verified = self._try_tool_verification(claim)

            if is_verified:
                verified_count += 1
            else:
                flagged.append(claim)

        # Compute confidence
        total = len(claims)
        confidence = verified_count / total if total > 0 else 1.0

        # Log
        if episode:
            episode.log_step(
                "self_check",
                f"Verification: {verified_count}/{total} claims verified, "
                f"{len(flagged)} flagged, confidence={confidence:.2f}",
            )

        verified = len(flagged) == 0
        report_lines = [
            f"Verified {verified_count}/{total} factual claims.",
        ]
        if flagged:
            report_lines.append("Flagged claims:")
            for f in flagged:
                report_lines.append(f"  - [unverified] {f}")

        return VerificationResult(
            verified=verified,
            confidence=round(confidence, 2),
            flagged_claims=flagged,
            report="\n".join(report_lines),
        )

    # ── Claim extraction ─────────────────────────────────────────────

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from the answer.

        Uses LLM if available; otherwise falls back to sentence-level
        heuristic extraction.
        """
        if self.llm is not None:
            try:
                return self._extract_claims_llm(answer)
            except Exception as exc:
                logger.warning("LLM claim extraction failed: %s", exc)

        return self._extract_claims_heuristic(answer)

    def _extract_claims_llm(self, answer: str) -> List[str]:
        """Use LLM to extract factual claims."""
        prompt = (
            "從以下回答中提取所有可驗證的事實性斷言，每行一個：\n\n"
            f"{answer}\n\n"
            "只列出事實性斷言，不要包含觀點或推測。每行一個。"
        )
        response = self.llm.generate(prompt, max_tokens=512)
        claims = [
            line.strip().lstrip("0123456789.-•) ")
            for line in response.strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
        return claims[:10]  # cap

    @staticmethod
    def _extract_claims_heuristic(answer: str) -> List[str]:
        """Extract sentences that look like factual claims."""
        # Split into sentences — avoid splitting on decimal points (e.g. 1.5)
        sentences = re.split(r"(?<!\d)[。.]\s*|[!！?？\n]", answer)
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) < 10:
                continue
            # Skip questions, opinions, hedging
            if any(kw in s for kw in ["？", "?", "我認為", "可能", "也許",
                                       "I think", "maybe", "perhaps"]):
                continue
            # Keep sentences with numbers, names, or definitive statements
            if any(c.isdigit() for c in s) or len(s) > 20:
                claims.append(s)
        return claims[:10]

    # ── Verification backends ────────────────────────────────────────

    def _check_against_knowledge(self, claim: str) -> bool:
        """Check if the claim is supported by KnowledgeStore."""
        if self.knowledge_store is None:
            return False
        try:
            return self.knowledge_store.has_knowledge(
                claim, threshold=self.confidence_threshold,
            )
        except Exception:
            return False

    @staticmethod
    def _check_against_trace(claim: str, trace_steps: List[str]) -> bool:
        """Check if the claim appears in recent tool observations."""
        claim_lower = claim.lower()
        for step in trace_steps:
            if claim_lower in step.lower():
                return True
            # Check for significant keyword overlap
            claim_words = set(claim_lower.split())
            step_words = set(step.lower().split())
            if len(claim_words) > 3:
                overlap = len(claim_words & step_words) / len(claim_words)
                if overlap > 0.5:
                    return True
        return False

    def _try_tool_verification(self, claim: str) -> bool:
        """Attempt to verify a claim using available tools."""
        if self.skill_registry is None:
            return False
        # Only attempt web_search for now
        tool = self.skill_registry.get("web_search")
        if tool is None:
            return False
        try:
            result = tool.execute(claim)
            # Simple heuristic: if search returns relevant content
            claim_words = set(claim.lower().split())
            result_words = set(result.lower().split())
            if len(claim_words) > 3:
                overlap = len(claim_words & result_words) / len(claim_words)
                return overlap > 0.3
        except Exception:
            pass
        return False

    def __repr__(self) -> str:
        mode = "LLM" if self.llm else "heuristic"
        return f"HallucinationGuard(mode={mode})"
