"""
Tests for the 3-stage HallucinationGuard.

Covers:
  - Stage 1: Claim extraction (LLM + heuristic)
  - Stage 2: Evidence matching (tool → KS external → KS reflexion)
  - Stage 3: Verdict determination (PASS / NEEDS_REVISION / UNRELIABLE)
  - Contradiction detection
  - Full pipeline integration
  - Backward compatibility
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning.compound_result import (
    ClaimVerification,
    VerificationResult,
    VERDICT_PASS,
    VERDICT_NEEDS_REVISION,
    VERDICT_UNRELIABLE,
)
from reasoning.hallucination_guard import (
    HallucinationGuard,
    STATUS_VERIFIED,
    STATUS_INFERRED,
    STATUS_UNVERIFIED,
    STATUS_CONFLICT,
)


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_ks_entry(content, source="web", confidence=0.8):
    """Create a mock KnowledgeStore entry."""
    entry = MagicMock()
    entry.content = content
    entry.source = source
    entry.confidence = confidence
    entry.query = f"query for {content[:20]}"
    return entry


def _make_ks(entries=None):
    """Create a mock KnowledgeStore."""
    ks = MagicMock()
    ks.search = MagicMock(return_value=entries or [])
    return ks


def _make_llm(response="claim 1\nclaim 2"):
    llm = MagicMock()
    llm.generate = MagicMock(return_value=response)
    return llm


# ═══════════════════════════════════════════════════════════════════
#  Tests — Stage 1: Claim Extraction
# ═══════════════════════════════════════════════════════════════════

class TestClaimExtraction:

    def test_heuristic_extracts_numeric_claims(self):
        """Heuristic extraction finds sentences with numbers."""
        claims = HallucinationGuard._extract_claims_heuristic(
            "The Earth is approximately 1.5 million km from the Sun. "
            "Water boils at 100 degrees Celsius at sea level."
        )
        assert len(claims) >= 1
        assert any("1.5" in c for c in claims)

    def test_heuristic_skips_questions(self):
        """Questions are not extracted as claims."""
        claims = HallucinationGuard._extract_claims_heuristic(
            "Is the sky blue? What about the Earth?"
        )
        assert len(claims) == 0

    def test_heuristic_skips_opinions(self):
        """Subjective opinions are skipped."""
        claims = HallucinationGuard._extract_claims_heuristic(
            "我認為這是一個好的方法。也許我們應該試試看。"
        )
        assert len(claims) == 0

    def test_heuristic_skips_short_text(self):
        """Short fragments are skipped."""
        claims = HallucinationGuard._extract_claims_heuristic("OK")
        assert len(claims) == 0

    def test_heuristic_caps_at_10(self):
        """Output is capped at 10 claims."""
        text = ". ".join([f"Claim number {i} is about a fact" for i in range(20)])
        claims = HallucinationGuard._extract_claims_heuristic(text)
        assert len(claims) <= 10

    def test_llm_extraction_used_when_available(self):
        """LLM extraction is preferred over heuristic."""
        llm = _make_llm("Python was created in 1991\nGuido van Rossum built it")
        guard = HallucinationGuard(llm=llm)
        claims = guard._extract_claims("Python was created by Guido.")
        assert len(claims) == 2
        llm.generate.assert_called_once()

    def test_llm_failure_falls_back_to_heuristic(self):
        """LLM failure → heuristic fallback."""
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=RuntimeError("fail"))
        guard = HallucinationGuard(llm=llm)
        claims = guard._extract_claims(
            "地球距離太陽約 1.5 億公里。"
        )
        # Should not raise; uses heuristic
        assert isinstance(claims, list)


# ═══════════════════════════════════════════════════════════════════
#  Tests — Stage 2: Evidence Matching
# ═══════════════════════════════════════════════════════════════════

class TestEvidenceMatching:

    # ── Tool results ─────────────────────────────────────────────

    def test_exact_match_in_tool_results(self):
        """Exact substring match in tool results → verified."""
        guard = HallucinationGuard()
        cv = guard._match_evidence(
            "The answer is 42",
            ["Tool returned: The answer is 42"],
        )
        assert cv.status == STATUS_VERIFIED
        assert cv.evidence_source == "tool"
        assert cv.confidence >= 0.9

    def test_keyword_overlap_in_tool_results(self):
        """Good keyword overlap in tool results → verified."""
        guard = HallucinationGuard()
        cv = guard._match_evidence(
            "Python was created by Guido van Rossum in 1991",
            ["Python programming language by Guido van Rossum since 1991 release"],
        )
        assert cv.status in (STATUS_VERIFIED, STATUS_INFERRED)
        assert cv.evidence_source == "tool"

    def test_no_match_in_tool_results(self):
        """No match in tool results → falls through."""
        guard = HallucinationGuard()
        cv = guard._match_evidence(
            "Earth is the third planet",
            ["Tool returned: weather is sunny"],
        )
        assert cv.status == STATUS_UNVERIFIED

    def test_contradiction_in_tool_results(self):
        """Contradicting tool result → conflict."""
        guard = HallucinationGuard()
        cv = guard._match_evidence(
            "The result is true",
            ["The result is false, not true"],
        )
        assert cv.status == STATUS_CONFLICT
        assert cv.evidence_source == "tool"

    # ── KnowledgeStore — external ────────────────────────────────

    def test_ks_external_exact_match(self):
        """Exact match in KS web entries → verified."""
        ks = _make_ks([_make_ks_entry("Python was created by Guido in 1991", "web")])
        guard = HallucinationGuard(knowledge_store=ks)
        cv = guard._check_knowledge_store(
            "Python was created by Guido in 1991",
            source_filter=("web", "admin"),
        )
        assert cv is not None
        assert cv.status == STATUS_VERIFIED
        assert cv.evidence_source == "knowledge"

    def test_ks_external_keyword_overlap(self):
        """Keyword overlap in KS web entries → verified."""
        ks = _make_ks([_make_ks_entry(
            "Python programming language was developed by Guido van Rossum in 1991",
            "web",
        )])
        guard = HallucinationGuard(knowledge_store=ks)
        cv = guard._check_knowledge_store(
            "Python was created by Guido van Rossum in the year 1991",
            source_filter=("web", "admin"),
        )
        assert cv is not None
        assert cv.status == STATUS_VERIFIED
        assert cv.evidence_source == "knowledge"

    def test_ks_no_entries(self):
        """Empty KS → None."""
        ks = _make_ks([])
        guard = HallucinationGuard(knowledge_store=ks)
        cv = guard._check_knowledge_store(
            "some claim", source_filter=("web",),
        )
        assert cv is None

    def test_ks_source_filtering(self):
        """Entries not matching source filter are skipped."""
        ks = _make_ks([
            _make_ks_entry("Python facts", "reflexion"),  # wrong source
        ])
        guard = HallucinationGuard(knowledge_store=ks)
        cv = guard._check_knowledge_store(
            "Python facts", source_filter=("web", "admin"),
        )
        assert cv is None

    # ── KnowledgeStore — reflexion ───────────────────────────────

    def test_ks_reflexion_match(self):
        """Match in reflexion entries → inferred (lower trust)."""
        ks = _make_ks([_make_ks_entry("Strategy X works well", "reflexion")])
        guard = HallucinationGuard(knowledge_store=ks)
        cv = guard._check_knowledge_store(
            "Strategy X works well", source_filter=("reflexion",),
        )
        assert cv is not None
        assert cv.status == STATUS_INFERRED
        assert cv.evidence_source == "reflexion"
        assert cv.confidence < 0.85  # lower than external

    # ── Full cascade ─────────────────────────────────────────────

    def test_cascade_tool_first(self):
        """Tool evidence is preferred over KS."""
        ks = _make_ks([_make_ks_entry("claim text here", "web")])
        guard = HallucinationGuard(knowledge_store=ks)
        cv = guard._match_evidence(
            "claim text here",
            ["claim text here"],  # exact match in tool results
        )
        assert cv.evidence_source == "tool"

    def test_cascade_ks_external_second(self):
        """When no tool match, KS external is checked."""
        ks = _make_ks([_make_ks_entry("the answer is 42", "web")])
        guard = HallucinationGuard(knowledge_store=ks)
        cv = guard._match_evidence(
            "the answer is 42",
            [],  # no tool results
        )
        assert cv.evidence_source == "knowledge"

    def test_cascade_ks_reflexion_third(self):
        """When no tool or external KS match, reflexion is checked."""
        ks = _make_ks([_make_ks_entry("the answer is 42", "reflexion")])
        guard = HallucinationGuard(knowledge_store=ks)
        cv = guard._match_evidence("the answer is 42", [])
        assert cv.evidence_source == "reflexion"


# ═══════════════════════════════════════════════════════════════════
#  Tests — Contradiction Detection
# ═══════════════════════════════════════════════════════════════════

class TestContradictionDetection:

    def test_negation_true_false(self):
        assert HallucinationGuard._texts_contradict(
            "the result is true",
            "the result is false",
        ) is True

    def test_chinese_negation(self):
        assert HallucinationGuard._texts_contradict(
            "這個方法不是正確的",
            "這個方法是正確的",
        ) is True

    def test_no_contradiction(self):
        assert HallucinationGuard._texts_contradict(
            "Python is great",
            "Python was created in 1991",
        ) is False

    def test_double_negation_no_contradiction(self):
        """If both texts have the negation, no contradiction."""
        assert HallucinationGuard._texts_contradict(
            "This isn't correct",
            "This isn't valid",
        ) is False


# ═══════════════════════════════════════════════════════════════════
#  Tests — Stage 3: Verdict Determination
# ═══════════════════════════════════════════════════════════════════

class TestVerdictDetermination:

    def test_all_verified_pass(self):
        """All verified → PASS."""
        guard = HallucinationGuard()
        claims = [
            ClaimVerification(claim="a", status=STATUS_VERIFIED, confidence=0.9),
            ClaimVerification(claim="b", status=STATUS_VERIFIED, confidence=0.85),
        ]
        assert guard._determine_verdict(claims) == VERDICT_PASS

    def test_all_inferred_pass(self):
        """All inferred → PASS."""
        guard = HallucinationGuard()
        claims = [
            ClaimVerification(claim="a", status=STATUS_INFERRED, confidence=0.7),
        ]
        assert guard._determine_verdict(claims) == VERDICT_PASS

    def test_mixed_verified_inferred_pass(self):
        """Mix of verified + inferred → PASS."""
        guard = HallucinationGuard()
        claims = [
            ClaimVerification(claim="a", status=STATUS_VERIFIED, confidence=0.9),
            ClaimVerification(claim="b", status=STATUS_INFERRED, confidence=0.6),
        ]
        assert guard._determine_verdict(claims) == VERDICT_PASS

    def test_some_unverified_needs_revision(self):
        """Some unverified (≤50%) → NEEDS_REVISION."""
        guard = HallucinationGuard()
        claims = [
            ClaimVerification(claim="a", status=STATUS_VERIFIED, confidence=0.9),
            ClaimVerification(claim="b", status=STATUS_VERIFIED, confidence=0.85),
            ClaimVerification(claim="c", status=STATUS_UNVERIFIED, confidence=0.0),
        ]
        assert guard._determine_verdict(claims) == VERDICT_NEEDS_REVISION

    def test_many_unverified_unreliable(self):
        """More than 50% unverified → UNRELIABLE."""
        guard = HallucinationGuard()
        claims = [
            ClaimVerification(claim="a", status=STATUS_UNVERIFIED, confidence=0.0),
            ClaimVerification(claim="b", status=STATUS_UNVERIFIED, confidence=0.0),
            ClaimVerification(claim="c", status=STATUS_VERIFIED, confidence=0.9),
        ]
        assert guard._determine_verdict(claims) == VERDICT_UNRELIABLE

    def test_any_conflict_unreliable(self):
        """Any conflict → UNRELIABLE."""
        guard = HallucinationGuard()
        claims = [
            ClaimVerification(claim="a", status=STATUS_VERIFIED, confidence=0.9),
            ClaimVerification(claim="b", status=STATUS_CONFLICT, confidence=0.1),
        ]
        assert guard._determine_verdict(claims) == VERDICT_UNRELIABLE

    def test_empty_claims_pass(self):
        """Empty claims → PASS."""
        guard = HallucinationGuard()
        assert guard._determine_verdict([]) == VERDICT_PASS


# ═══════════════════════════════════════════════════════════════════
#  Tests — Full Pipeline
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:

    def test_no_claims_accepted(self):
        """Short answer with no claims → PASS."""
        guard = HallucinationGuard()
        result = guard.verify("OK", "task")
        assert result.verdict == VERDICT_PASS
        assert result.verified is True
        assert result.confidence >= 0.9
        assert result.hallucination_score == 0.0

    def test_verified_claims_pass(self):
        """All claims verified by tool results → PASS."""
        guard = HallucinationGuard()
        result = guard.verify(
            "The answer is 42 and the result is confirmed",
            "What is the answer?",
            tool_results=[
                "Tool returned: The answer is 42 and the result is confirmed",
            ],
        )
        assert result.verdict == VERDICT_PASS
        assert result.verified is True
        assert len(result.flagged_claims) == 0

    def test_unverified_claims_flagged(self):
        """Claims not found anywhere → flagged."""
        guard = HallucinationGuard()
        result = guard.verify(
            "Pluto has 37 confirmed moons orbiting it in 2024",
            "How many moons does Pluto have?",
            tool_results=[],
        )
        assert len(result.flagged_claims) >= 1
        assert result.verdict in (VERDICT_NEEDS_REVISION, VERDICT_UNRELIABLE)

    def test_ks_verified_claims_pass(self):
        """Claims verified by KnowledgeStore → PASS."""
        ks = _make_ks([
            _make_ks_entry("Python was created in 1991 by Guido van Rossum", "web"),
        ])
        guard = HallucinationGuard(knowledge_store=ks)
        result = guard.verify(
            "Python was created in 1991 by Guido van Rossum",
            "When was Python created?",
        )
        assert result.verdict == VERDICT_PASS
        assert result.verified is True

    def test_result_has_all_fields(self):
        """VerificationResult has all expected fields."""
        guard = HallucinationGuard()
        result = guard.verify("test answer 12345", "task")
        assert hasattr(result, "verdict")
        assert hasattr(result, "claims")
        assert hasattr(result, "hallucination_score")
        assert hasattr(result, "verified")
        assert hasattr(result, "confidence")
        assert hasattr(result, "flagged_claims")
        assert hasattr(result, "report")

    def test_episode_logging(self):
        """Verification logs to episode."""
        ep = MagicMock()
        guard = HallucinationGuard()
        guard.verify("OK", "task", episode=ep)
        ep.log_step.assert_called()

    def test_llm_extraction_with_full_pipeline(self):
        """LLM extraction integrates into full pipeline."""
        llm = _make_llm(
            "Python was created in 1991\nGuido van Rossum built it"
        )
        guard = HallucinationGuard(llm=llm)
        result = guard.verify(
            "Python was created in 1991 by Guido van Rossum.",
            "about Python",
            tool_results=[
                "Python 1991 Guido van Rossum creator",
            ],
        )
        assert isinstance(result, VerificationResult)

    def test_conflict_in_tool_results(self):
        """Contradicting tool result → UNRELIABLE."""
        guard = HallucinationGuard()
        result = guard.verify(
            "The system supports multithreading and is true",
            "Does it support multithreading?",
            tool_results=[
                "The system does not support multithreading and is false",
            ],
        )
        assert result.verdict == VERDICT_UNRELIABLE
        assert any(
            c.status == STATUS_CONFLICT for c in result.claims
        )


# ═══════════════════════════════════════════════════════════════════
#  Tests — Backward Compatibility
# ═══════════════════════════════════════════════════════════════════

class TestBackwardCompat:

    def test_check_against_trace_match(self):
        """Legacy method: claim found in trace → True."""
        result = HallucinationGuard._check_against_trace(
            "The answer is 42",
            ["Tool returned: The answer is 42"],
        )
        assert result is True

    def test_check_against_trace_miss(self):
        """Legacy method: claim not in trace → False."""
        result = HallucinationGuard._check_against_trace(
            "Earth is the third planet",
            ["Tool returned: weather is sunny"],
        )
        assert result is False

    def test_verification_result_has_legacy_fields(self):
        """VerificationResult still has verified/confidence/flagged_claims."""
        guard = HallucinationGuard()
        result = guard.verify("OK", "task")
        assert isinstance(result.verified, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.flagged_claims, list)
        assert isinstance(result.report, str)


# ═══════════════════════════════════════════════════════════════════
#  Tests — ClaimVerification Dataclass
# ═══════════════════════════════════════════════════════════════════

class TestClaimVerification:

    def test_defaults(self):
        cv = ClaimVerification()
        assert cv.claim == ""
        assert cv.status == "unverified"
        assert cv.evidence_source is None
        assert cv.evidence_text is None
        assert cv.confidence == 0.0

    def test_custom_values(self):
        cv = ClaimVerification(
            claim="fact",
            status=STATUS_VERIFIED,
            evidence_source="tool",
            evidence_text="some evidence",
            confidence=0.95,
        )
        assert cv.claim == "fact"
        assert cv.status == STATUS_VERIFIED
        assert cv.confidence == 0.95
