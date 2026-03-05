"""
HallucinationGuard — 3-stage Answer Verification Gate (Phase ③).

Designed for small models (Mistral-7B) where hallucination rates are
significantly higher than frontier models.

Pipeline::

    Stage 1  Claim Extraction    → regex + LLM factual assertion extraction
    Stage 2  Evidence Matching   → tool_results → KnowledgeStore → reflexion
    Stage 3  Verdict & Action    → PASS / NEEDS_REVISION / UNRELIABLE

References
~~~~~~~~~~
- Chain-of-Knowledge (Li et al., ICLR 2024, arXiv:2305.13269):
  multi-source knowledge cross-validation.
- "LLMs Cannot Self-Correct Reasoning Yet" (Huang et al., ICLR 2024,
  arXiv:2310.01798): external signals needed for self-verification.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional

from reasoning.compound_result import (
    ClaimVerification,
    VerificationResult,
    VERDICT_PASS,
    VERDICT_NEEDS_REVISION,
    VERDICT_UNRELIABLE,
)

if TYPE_CHECKING:
    from core.llm_interface import BaseLLM
    from memory.episodic_log import EpisodicLog
    from rag.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

# ── Status constants ─────────────────────────────────────────────────
STATUS_VERIFIED = "verified"
STATUS_INFERRED = "inferred"
STATUS_UNVERIFIED = "unverified"
STATUS_CONFLICT = "conflict"

# ── Default thresholds ───────────────────────────────────────────────
_DEFAULT_CONFLICT_THRESHOLD = 0.7
_KEYWORD_OVERLAP_VERIFIED = 0.5
_KEYWORD_OVERLAP_INFERRED = 0.3
_MAX_CLAIMS = 10
_UNVERIFIED_RATIO_UNRELIABLE = 0.5  # > 50% unverified → UNRELIABLE


class HallucinationGuard:
    """Three-stage answer verification for small-LLM agents.

    Stage 1 — **Claim Extraction**: extracts key factual assertions.
    Stage 2 — **Evidence Matching**: checks each claim against tool
              results, KnowledgeStore (web/admin), then reflexion.
    Stage 3 — **Verdict & Action**: determines overall reliability.

    Args:
        llm:                 LLM backend for claim extraction.
        knowledge_store:     KnowledgeStore for cross-validation.
        tools:               Optional callable dict for live verification
                             (e.g. ``{"web_search": callable}``).
        conflict_threshold:  Keyword overlap ratio to detect conflict.
    """

    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,
        knowledge_store: Optional["KnowledgeStore"] = None,
        tools: Optional[Dict[str, object]] = None,
        conflict_threshold: float = _DEFAULT_CONFLICT_THRESHOLD,
    ) -> None:
        self.llm = llm
        self.knowledge_store = knowledge_store
        self.tools = tools or {}
        self.conflict_threshold = conflict_threshold

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def verify(
        self,
        answer: str,
        task: str,
        tool_results: Optional[List[str]] = None,
        trace: Optional[List[str]] = None,
        episode: Optional["EpisodicLog"] = None,
    ) -> VerificationResult:
        """Run the full 3-stage verification pipeline.

        Args:
            answer:       The candidate answer text.
            task:         Original task description.
            tool_results: Tool output strings from the current episode.
            trace:        Additional trace/observation strings.
            episode:      Optional episodic log for recording steps.

        Returns:
            :class:`VerificationResult` with verdict and per-claim details.
        """
        all_tool_evidence = list(tool_results or []) + list(trace or [])

        # ── Stage 1: Claim Extraction ────────────────────────────
        claims = self._extract_claims(answer)

        if not claims:
            if episode:
                episode.log_step(
                    "self_check", "No factual claims to verify",
                )
            return VerificationResult(
                verdict=VERDICT_PASS,
                verified=True,
                confidence=0.9,
                hallucination_score=0.0,
                report="No factual claims extracted; answer accepted.",
            )

        # ── Stage 2: Evidence Matching ───────────────────────────
        claim_results: List[ClaimVerification] = []
        for claim_text in claims:
            cv = self._match_evidence(claim_text, all_tool_evidence)
            claim_results.append(cv)

        # ── Stage 3: Verdict & Action ────────────────────────────
        verdict = self._determine_verdict(claim_results)

        # Compute backward-compat fields
        verified_count = sum(
            1 for c in claim_results
            if c.status in (STATUS_VERIFIED, STATUS_INFERRED)
        )
        total = len(claim_results)
        confidence = verified_count / total if total > 0 else 1.0
        hallucination_score = 1.0 - confidence

        flagged = [
            c.claim for c in claim_results
            if c.status in (STATUS_UNVERIFIED, STATUS_CONFLICT)
        ]

        # Build report
        report_lines = [
            f"Verdict: {verdict}",
            f"Claims: {verified_count}/{total} verified, "
            f"{len(flagged)} flagged",
        ]
        for cv in claim_results:
            tag = cv.status.upper()
            src = f" ({cv.evidence_source})" if cv.evidence_source else ""
            report_lines.append(f"  [{tag}]{src} {cv.claim[:80]}")
        if flagged:
            report_lines.append("Flagged claims:")
            for f in flagged:
                report_lines.append(f"  - [unverified] {f}")

        # Log to episodic log
        if episode:
            episode.log_step(
                "self_check",
                f"Verification: verdict={verdict}, "
                f"{verified_count}/{total} verified, "
                f"hallucination_score={hallucination_score:.2f}",
            )

        logger.info(
            "HallucinationGuard: verdict=%s, verified=%d/%d, "
            "hallucination=%.2f",
            verdict, verified_count, total, hallucination_score,
        )

        return VerificationResult(
            verdict=verdict,
            claims=claim_results,
            revised_answer=None,
            hallucination_score=round(hallucination_score, 2),
            verified=(verdict == VERDICT_PASS),
            confidence=round(confidence, 2),
            flagged_claims=flagged,
            report="\n".join(report_lines),
        )

    # ═══════════════════════════════════════════════════════════════
    #  Stage 1 — Claim Extraction
    # ═══════════════════════════════════════════════════════════════

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract verifiable factual assertions from the answer.

        Uses LLM if available; falls back to sentence-level heuristics.
        Skips subjective opinions and hedged statements.
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
        return claims[:_MAX_CLAIMS]

    @staticmethod
    def _extract_claims_heuristic(answer: str) -> List[str]:
        """Extract sentences that look like factual claims.

        Heuristics:
          - Split on sentence boundaries (handles decimal numbers).
          - Skip questions, opinions, hedging.
          - Keep statements with numbers, names, or longer assertions.
        """
        # Split into sentences — avoid splitting on decimal points
        sentences = re.split(r"(?<!\d)[。.]\s*|[!！?？\n]", answer)
        claims: List[str] = []
        for s in sentences:
            s = s.strip()
            if len(s) < 10:
                continue
            # Skip questions, opinions, hedging
            if any(kw in s for kw in [
                "？", "?", "我認為", "可能", "也許",
                "I think", "maybe", "perhaps",
            ]):
                continue
            # Keep sentences with numbers, names, or definitive statements
            if any(c.isdigit() for c in s) or len(s) > 20:
                claims.append(s)
        return claims[:_MAX_CLAIMS]

    # ═══════════════════════════════════════════════════════════════
    #  Stage 2 — Evidence Matching
    # ═══════════════════════════════════════════════════════════════

    def _match_evidence(
        self,
        claim: str,
        tool_results: List[str],
    ) -> ClaimVerification:
        """Find the best evidence for a single claim.

        Priority:
          1. tool_results    (highest trust — direct observation)
          2. knowledge_store (web/admin source — factual)
          3. reflexion_knowledge (agent's own inference — lower trust)
          4. no_evidence

        Also detects conflicts (contradicting evidence).
        """
        # 1. Check tool results (most reliable)
        result = self._check_tool_results(claim, tool_results)
        if result is not None:
            return result

        # 2. Check KnowledgeStore — external sources (web/admin)
        result = self._check_knowledge_store(
            claim, source_filter=("web", "admin"),
        )
        if result is not None:
            return result

        # 3. Check KnowledgeStore — reflexion sources (lower trust)
        result = self._check_knowledge_store(
            claim, source_filter=("reflexion",),
        )
        if result is not None:
            return result

        # 4. No evidence found
        return ClaimVerification(
            claim=claim,
            status=STATUS_UNVERIFIED,
            evidence_source=None,
            evidence_text=None,
            confidence=0.0,
        )

    def _check_tool_results(
        self,
        claim: str,
        tool_results: List[str],
    ) -> Optional[ClaimVerification]:
        """Check claim against tool output strings.

        Returns ClaimVerification if evidence found, else None.
        """
        if not tool_results:
            return None

        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())

        for result_text in tool_results:
            result_lower = result_text.lower()

            # Contradiction check FIRST (before substring match)
            # "claim is true" inside "claim is false, not true" must
            # be caught as conflict, not verified by substring.
            if self._texts_contradict(claim, result_text):
                return ClaimVerification(
                    claim=claim,
                    status=STATUS_CONFLICT,
                    evidence_source="tool",
                    evidence_text=result_text[:200],
                    confidence=0.1,
                )

            # Exact substring match → verified
            if claim_lower in result_lower:
                return ClaimVerification(
                    claim=claim,
                    status=STATUS_VERIFIED,
                    evidence_source="tool",
                    evidence_text=result_text[:200],
                    confidence=0.95,
                )

            # Keyword overlap → verified or inferred
            if len(claim_words) > 3:
                result_words = set(result_lower.split())
                overlap = (
                    len(claim_words & result_words) / len(claim_words)
                )
                if overlap >= _KEYWORD_OVERLAP_VERIFIED:
                    return ClaimVerification(
                        claim=claim,
                        status=STATUS_VERIFIED,
                        evidence_source="tool",
                        evidence_text=result_text[:200],
                        confidence=round(0.7 + overlap * 0.2, 2),
                    )
                if overlap >= _KEYWORD_OVERLAP_INFERRED:
                    return ClaimVerification(
                        claim=claim,
                        status=STATUS_INFERRED,
                        evidence_source="tool",
                        evidence_text=result_text[:200],
                        confidence=round(0.4 + overlap * 0.3, 2),
                    )

        return None

    def _check_knowledge_store(
        self,
        claim: str,
        source_filter: tuple[str, ...] = ("web", "admin"),
    ) -> Optional[ClaimVerification]:
        """Check claim against KnowledgeStore entries.

        Args:
            claim:         The claim to verify.
            source_filter: Only consider entries from these sources.

        Returns:
            ClaimVerification if evidence found, else None.
        """
        if self.knowledge_store is None:
            return None

        try:
            entries = self.knowledge_store.search(claim, top_k=3)
        except Exception:
            return None

        if not entries:
            return None

        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())
        is_reflexion = "reflexion" in source_filter

        for entry in entries:
            entry_source = getattr(entry, "source", "web")
            if entry_source not in source_filter:
                continue

            entry_content = getattr(entry, "content", "")
            entry_lower = entry_content.lower()

            # Check for contradiction
            if self._texts_contradict(claim, entry_content):
                return ClaimVerification(
                    claim=claim,
                    status=STATUS_CONFLICT,
                    evidence_source="reflexion" if is_reflexion else "knowledge",
                    evidence_text=entry_content[:200],
                    confidence=0.1,
                )

            # Exact substring match
            if claim_lower in entry_lower:
                conf = 0.7 if is_reflexion else 0.85
                return ClaimVerification(
                    claim=claim,
                    status=STATUS_VERIFIED if not is_reflexion else STATUS_INFERRED,
                    evidence_source="reflexion" if is_reflexion else "knowledge",
                    evidence_text=entry_content[:200],
                    confidence=conf,
                )

            # Keyword overlap
            if len(claim_words) > 3:
                entry_words = set(entry_lower.split())
                overlap = (
                    len(claim_words & entry_words) / len(claim_words)
                )
                if overlap >= _KEYWORD_OVERLAP_VERIFIED:
                    conf = 0.6 if is_reflexion else 0.75
                    status = (
                        STATUS_INFERRED if is_reflexion
                        else STATUS_VERIFIED
                    )
                    return ClaimVerification(
                        claim=claim,
                        status=status,
                        evidence_source="reflexion" if is_reflexion else "knowledge",
                        evidence_text=entry_content[:200],
                        confidence=conf,
                    )

        return None

    # ── Contradiction detection ──────────────────────────────────

    @staticmethod
    def _texts_contradict(text_a: str, text_b: str) -> bool:
        """Detect obvious contradictions between two texts.

        Uses negation pattern heuristic. A production system would
        use NLI or embedding cosine distance.
        """
        a = text_a.lower()
        b = text_b.lower()

        negation_pairs = [
            ("不是", "是"), ("沒有", "有"), ("不能", "能"),
            ("false", "true"), ("incorrect", "correct"),
            ("no", "yes"), ("isn't", "is"), ("不對", "對"),
            ("不支持", "支持"), ("不包含", "包含"),
        ]
        for neg, pos in negation_pairs:
            if neg in a and pos in b and neg not in b:
                return True
            if neg in b and pos in a and neg not in a:
                return True
        return False

    # ═══════════════════════════════════════════════════════════════
    #  Stage 3 — Verdict & Action
    # ═══════════════════════════════════════════════════════════════

    def _determine_verdict(
        self,
        claims: List[ClaimVerification],
    ) -> str:
        """Determine overall verdict from per-claim results.

        Rules:
          - Any ``conflict`` → UNRELIABLE.
          - More than 50% ``unverified`` → UNRELIABLE.
          - Any ``unverified`` (but ≤50%) → NEEDS_REVISION.
          - All ``verified`` or ``inferred`` → PASS.
        """
        if not claims:
            return VERDICT_PASS

        total = len(claims)
        conflict_count = sum(
            1 for c in claims if c.status == STATUS_CONFLICT
        )
        unverified_count = sum(
            1 for c in claims if c.status == STATUS_UNVERIFIED
        )

        if conflict_count > 0:
            return VERDICT_UNRELIABLE

        if total > 0 and unverified_count / total > _UNVERIFIED_RATIO_UNRELIABLE:
            return VERDICT_UNRELIABLE

        if unverified_count > 0:
            return VERDICT_NEEDS_REVISION

        return VERDICT_PASS

    # ── Legacy compatibility ──────────────────────────────────────

    @staticmethod
    def _check_against_trace(claim: str, trace_steps: List[str]) -> bool:
        """Legacy: check if claim text appears in trace observations.

        Kept for backward compatibility with CompoundReasoner tests.
        """
        claim_lower = claim.lower()
        for step in trace_steps:
            if claim_lower in step.lower():
                return True
            claim_words = set(claim_lower.split())
            step_words = set(step.lower().split())
            if len(claim_words) > 3:
                overlap = len(claim_words & step_words) / len(claim_words)
                if overlap > 0.5:
                    return True
        return False

    def __repr__(self) -> str:
        mode = "LLM" if self.llm else "heuristic"
        ks = "yes" if self.knowledge_store else "no"
        return (
            f"HallucinationGuard(mode={mode}, ks={ks}, "
            f"tools={list(self.tools.keys())})"
        )
