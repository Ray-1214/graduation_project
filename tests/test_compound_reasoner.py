"""
Tests for CompoundReasoner and its sub-components.

Covers:
  - Phase ① Structured Thinking (output parsing)
  - Phase ② Adaptive Execution (all 4 modes)
  - Think Step logging
  - Grounded Self-Check
  - Phase ③ Verification Gate
  - Backward compatibility (force_strategy)
  - ContextAssembler
  - HallucinationGuard
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.episodic_log import EpisodicLog
from reasoning.compound_result import (
    CompoundResult,
    SelfCheckResult,
    ThinkingResult,
    VerificationResult,
)
from reasoning.context_assembler import ContextAssembler
from reasoning.hallucination_guard import HallucinationGuard
from reasoning.compound_reasoner import CompoundReasoner


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

def _make_llm(responses=None):
    """Build a mock LLM that returns canned responses in order."""
    llm = MagicMock()
    if responses is None:
        responses = ["mock response"]
    side = list(responses)
    llm.generate = MagicMock(side_effect=side)
    return llm


def _make_cot(answer="CoT answer"):
    cot = MagicMock()
    cot.run = MagicMock(return_value=answer)
    return cot


def _make_tot(answer="ToT answer"):
    tot = MagicMock()
    tot.search = MagicMock(return_value=answer)
    return tot


def _make_react(answer="ReAct answer"):
    react = MagicMock()
    react.run = MagicMock(return_value=answer)
    return react


def _make_reflexion():
    return MagicMock()


def _make_guard(verified=True, confidence=0.95):
    guard = MagicMock()
    guard.verify = MagicMock(return_value=VerificationResult(
        verified=verified, confidence=confidence,
    ))
    return guard


def _make_assembler():
    return ContextAssembler()


def _make_skills():
    skills = MagicMock()
    skills.list_descriptions = MagicMock(return_value="- web_search: search")
    skills.execute = MagicMock(return_value="tool result: 42")
    return skills


def _make_episode():
    return EpisodicLog(task="test", strategy="compound")


def _make_reasoner(llm=None, **overrides):
    """Build a CompoundReasoner with sensible mocks."""
    if llm is None:
        llm = _make_llm(["mock"] * 20)
    kw = dict(
        llm=llm,
        prompt_builder=MagicMock(),
        skill_registry=_make_skills(),
        cot=_make_cot(),
        tot=_make_tot(),
        react=_make_react(),
        reflexion=_make_reflexion(),
        hallucination_guard=_make_guard(),
        context_assembler=_make_assembler(),
        knowledge_store=None,
        check_interval=3,
        max_steps=10,
    )
    kw.update(overrides)
    return CompoundReasoner(**kw)


# ═══════════════════════════════════════════════════════════════════
#  Tests — Phase ① Structured Thinking
# ═══════════════════════════════════════════════════════════════════

class TestStructuredThinking:

    def test_parse_valid_analysis(self):
        """Well-formatted LLM output is correctly parsed."""
        raw = (
            "1. 任務類型：推理計算\n"
            "2. 預估複雜度：complex\n"
            "3. 需要的工具：web_search, calculator\n"
            "4. 推薦策略路線：react_cot\n"
            "5. 子目標拆解：[搜尋資料, 計算結果, 驗證答案]\n"
        )
        cr = _make_reasoner()
        result = cr._parse_thinking(raw)

        assert result.task_type == "推理計算"
        assert result.complexity == "complex"
        assert result.tools_needed == ["web_search", "calculator"]
        assert result.recommended_strategy == "react_cot"
        assert result.sub_goals == ["搜尋資料", "計算結果", "驗證答案"]

    def test_parse_empty_returns_defaults(self):
        """Empty LLM output returns default ThinkingResult."""
        cr = _make_reasoner()
        result = cr._parse_thinking("")

        assert result.task_type == "unknown"
        assert result.complexity == "moderate"
        assert result.recommended_strategy == "cot_only"
        assert result.sub_goals == []

    def test_parse_no_tools(self):
        """When tools = 'none', tools_needed is empty."""
        raw = "3. 需要的工具：none\n"
        cr = _make_reasoner()
        result = cr._parse_thinking(raw)
        assert result.tools_needed == []

    def test_structured_thinking_logged(self):
        """Phase ① output is logged as 'thinking_plan'."""
        analysis = (
            "1. 任務類型：事實查詢\n"
            "2. 預估複雜度：simple\n"
            "3. 需要的工具：none\n"
            "4. 推薦策略路線：cot_only\n"
            "5. 子目標拆解：[回答問題]\n"
        )
        llm = _make_llm([analysis])
        cr = _make_reasoner(llm=llm)
        ep = _make_episode()

        cr._structured_thinking("test task", ep)

        step_types = [s.step_type for s in ep.steps]
        assert "thinking_plan" in step_types


# ═══════════════════════════════════════════════════════════════════
#  Tests — Phase ② Adaptive Execution
# ═══════════════════════════════════════════════════════════════════

class TestAdaptiveExecution:

    def test_cot_only_delegates_to_cot(self):
        """'cot_only' strategy → delegates to CoT engine."""
        cr = _make_reasoner()
        ep = _make_episode()
        plan = ThinkingResult(recommended_strategy="cot_only")

        answer, steps, thinks, checks = cr._adaptive_execute("task", plan, ep)

        cr.cot.run.assert_called_once()
        assert answer == "CoT answer"
        assert steps == 1
        assert thinks == 0

    def test_react_cot_runs_react_loop(self):
        """'react_cot' runs the internal ReAct loop."""
        # LLM returns a Finish on first thought
        llm = _make_llm(["Thought: done\nFinish[42]"])
        cr = _make_reasoner(llm=llm)
        ep = _make_episode()
        plan = ThinkingResult(recommended_strategy="react_cot")

        answer, steps, thinks, checks = cr._adaptive_execute("task", plan, ep)

        assert answer == "42"
        assert steps >= 1

    def test_react_tot_uses_tot_on_later_steps(self):
        """'react_tot' invokes ToT search after accumulating steps."""
        responses = [
            # Step 1: action
            "Thought: step1\nAction[web_search]: query",
            # Think step after observation
            "Analysis: result is good",
            # Step 2 uses ToT (returns from tot mock)
            # Then step 3 finishes
            "Thought: done\nFinish[final answer]",
            # Extra think_step
            "Think analysis",
        ]
        llm = _make_llm(responses * 3)
        cr = _make_reasoner(llm=llm)
        cr.tot.search = MagicMock(return_value="ToT: best path\nFinish[tot-answer]")
        ep = _make_episode()
        plan = ThinkingResult(recommended_strategy="react_tot")

        answer, steps, thinks, checks = cr._adaptive_execute("task", plan, ep)
        # Should finish in some steps
        assert answer is not None

    def test_full_compound_enables_self_check(self):
        """'full_compound' runs self-checks at intervals."""
        # Make it run a few steps then finish
        responses = []
        for i in range(5):
            responses.append(f"Thought: step{i}\nAction[web_search]: q{i}")
            responses.append(f"Think: analysis {i}")
        responses.append("Thought: done\nFinish[result]")
        responses.append("Think: final")

        llm = _make_llm(responses * 2)
        cr = _make_reasoner(llm=llm, check_interval=2)
        ep = _make_episode()
        plan = ThinkingResult(
            recommended_strategy="full_compound",
            sub_goals=["搜尋", "計算"],
        )

        answer, steps, thinks, checks = cr._adaptive_execute("task", plan, ep)

        # At least one self-check should have run
        step_types = [s.step_type for s in ep.steps]
        assert "self_check" in step_types

    def test_strategy_switch_logged(self):
        """Strategy selection is logged as 'strategy_switch'."""
        cr = _make_reasoner()
        ep = _make_episode()
        plan = ThinkingResult(recommended_strategy="cot_only")

        cr._adaptive_execute("task", plan, ep)

        step_types = [s.step_type for s in ep.steps]
        assert "strategy_switch" in step_types


# ═══════════════════════════════════════════════════════════════════
#  Tests — Think Step
# ═══════════════════════════════════════════════════════════════════

class TestThinkStep:

    def test_think_step_logged_as_think(self):
        """Think step is recorded with type='think'."""
        llm = _make_llm(["1. 結果顯示42\n2. 一致\n3. 繼續"])
        cr = _make_reasoner(llm=llm)
        ep = _make_episode()
        plan = ThinkingResult(sub_goals=["goal1"])

        cr._think_step("tool returned 42", plan, ["prev step"], ep)

        step_types = [s.step_type for s in ep.steps]
        assert "think" in step_types

    def test_think_step_handles_llm_failure(self):
        """If LLM fails during think, a fallback message is used."""
        llm = _make_llm()
        llm.generate = MagicMock(side_effect=RuntimeError("LLM down"))
        cr = _make_reasoner(llm=llm)
        ep = _make_episode()

        result = cr._think_step("obs", ThinkingResult(), [], ep)
        assert "error" in result.lower()


# ═══════════════════════════════════════════════════════════════════
#  Tests — Grounded Self-Check
# ═══════════════════════════════════════════════════════════════════

class TestGroundedSelfCheck:

    def test_no_issues_passes(self):
        """Clean steps → passed=True, needs_replan=False."""
        cr = _make_reasoner()
        ep = _make_episode()
        plan = ThinkingResult(sub_goals=["搜尋資料"])
        steps = ["搜尋了資料", "Observation: 找到三篇文章"]

        result = cr._grounded_self_check("task", steps, plan, ep)

        assert result.passed is True
        assert result.needs_replan is False

    def test_contradiction_triggers_replan(self):
        """Contradictory observations trigger needs_replan."""
        cr = _make_reasoner()
        ep = _make_episode()
        plan = ThinkingResult()
        steps = [
            "結論是 true",
            "Observation: 答案是 false",
        ]

        result = cr._grounded_self_check("task", steps, plan, ep)

        # The negation heuristic should catch true/false contradiction
        assert result.needs_replan is True
        assert len(result.issues) > 0

    def test_progress_tracking(self):
        """Sub-goal completion is tracked."""
        cr = _make_reasoner()
        ep = _make_episode()
        plan = ThinkingResult(sub_goals=["search papers", "compute results"])
        steps = ["I will search papers now", "found three papers"]

        result = cr._grounded_self_check("task", steps, plan, ep)
        assert result.progress_pct > 0

    def test_self_check_logged(self):
        """Self-check is recorded with type='self_check'."""
        cr = _make_reasoner()
        ep = _make_episode()
        result = cr._grounded_self_check("t", [], ThinkingResult(), ep)

        step_types = [s.step_type for s in ep.steps]
        assert "self_check" in step_types


# ═══════════════════════════════════════════════════════════════════
#  Tests — Phase ③ Verification
# ═══════════════════════════════════════════════════════════════════

class TestVerificationGate:

    def test_verification_delegates_to_guard(self):
        """_verify_answer delegates to HallucinationGuard.verify."""
        guard = _make_guard(verified=True, confidence=0.9)
        cr = _make_reasoner(hallucination_guard=guard)
        ep = _make_episode()

        result = cr._verify_answer("answer", "task", [], ep)

        guard.verify.assert_called_once()
        assert result.verified is True

    def test_flagged_claims_tagged(self):
        """Flagged claims get [unverified] tag in final answer."""
        v = VerificationResult(
            flagged_claims=["Earth is flat"],
        )
        answer = "The answer is: Earth is flat."
        result = CompoundReasoner._apply_flags(answer, v)

        assert "[unverified] Earth is flat" in result

    def test_no_flags_unchanged(self):
        """Answer without flags is unchanged."""
        v = VerificationResult(flagged_claims=[])
        answer = "2 + 2 = 4"
        assert CompoundReasoner._apply_flags(answer, v) == answer


# ═══════════════════════════════════════════════════════════════════
#  Tests — Backward Compatibility
# ═══════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:

    def test_force_cot(self):
        """force_strategy='cot' → CoT directly, no Phase ①."""
        cr = _make_reasoner()
        ep = _make_episode()

        result = cr.run("task", ep, force_strategy="cot")

        assert isinstance(result, CompoundResult)
        assert result.answer == "CoT answer"
        assert result.thinking is None  # Phase ① skipped
        assert result.verification is None
        cr.cot.run.assert_called_once()

    def test_force_tot(self):
        """force_strategy='tot' → ToT search directly."""
        cr = _make_reasoner()
        ep = _make_episode()

        result = cr.run("task", ep, force_strategy="tot")

        assert result.answer == "ToT answer"
        cr.tot.search.assert_called_once()

    def test_force_react(self):
        """force_strategy='react' → ReAct loop directly."""
        cr = _make_reasoner()
        ep = _make_episode()

        result = cr.run("task", ep, force_strategy="react")

        assert result.answer == "ReAct answer"
        cr.react.run.assert_called_once()

    def test_force_unknown_falls_back_to_cot(self):
        """Unknown forced strategy → fallback to CoT."""
        cr = _make_reasoner()
        ep = _make_episode()

        result = cr.run("task", ep, force_strategy="unknown_strat")
        assert result.answer == "CoT answer"

    def test_forced_strategy_logged(self):
        """Forced strategy selection is logged."""
        cr = _make_reasoner()
        ep = _make_episode()

        cr.run("task", ep, force_strategy="cot")

        step_types = [s.step_type for s in ep.steps]
        assert "strategy_switch" in step_types
        contents = [s.content for s in ep.steps]
        assert any("Forced strategy: cot" in c for c in contents)


# ═══════════════════════════════════════════════════════════════════
#  Tests — Full Pipeline (run without force)
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:

    def test_full_run_cot_only(self):
        """Full pipeline with cot_only → answer + verification."""
        analysis = (
            "1. 任務類型：事實查詢\n"
            "2. 預估複雜度：simple\n"
            "3. 需要的工具：none\n"
            "4. 推薦策略路線：cot_only\n"
            "5. 子目標拆解：[回答問題]\n"
        )
        llm = _make_llm([analysis])
        cr = _make_reasoner(llm=llm)
        ep = _make_episode()

        result = cr.run("2+2=?", ep)

        assert isinstance(result, CompoundResult)
        assert result.answer == "CoT answer"
        assert result.thinking is not None
        assert result.thinking.recommended_strategy == "cot_only"
        assert result.verification is not None

    def test_full_run_react_cot_finish(self):
        """Full pipeline with react_cot strategy."""
        analysis = (
            "1. 任務類型：多步驟操作\n"
            "2. 預估複雜度：moderate\n"
            "3. 需要的工具：web_search\n"
            "4. 推薦策略路線：react_cot\n"
            "5. 子目標拆解：[搜尋, 彙整]\n"
        )
        # LLM calls: Phase①, react thought (finish), think step
        responses = [
            analysis,
            "Thought: I have enough info.\nFinish[the answer is 42]",
        ]
        llm = _make_llm(responses)
        cr = _make_reasoner(llm=llm)
        ep = _make_episode()

        result = cr.run("what is the answer?", ep)

        assert result.answer is not None
        assert result.strategy_used == "react_cot"
        assert result.steps_taken >= 1

    def test_compound_result_fields(self):
        """CompoundResult has all expected fields."""
        analysis = (
            "4. 推薦策略路線：cot_only\n"
        )
        llm = _make_llm([analysis])
        cr = _make_reasoner(llm=llm)
        ep = _make_episode()

        result = cr.run("task", ep)

        assert hasattr(result, "answer")
        assert hasattr(result, "confidence")
        assert hasattr(result, "strategy_used")
        assert hasattr(result, "thinking")
        assert hasattr(result, "verification")
        assert hasattr(result, "steps_taken")
        assert hasattr(result, "think_steps")
        assert hasattr(result, "self_checks")


# ═══════════════════════════════════════════════════════════════════
#  Tests — ContextAssembler
# ═══════════════════════════════════════════════════════════════════

class TestContextAssembler:

    def test_all_parts(self):
        """All parts combined in correct order."""
        ca = ContextAssembler()
        result = ca.assemble(
            task="question",
            skills_block="[skills]",
            lessons="[lessons]",
            rag_context="[rag]",
            thinking_plan="plan",
        )
        # Ordering: skills → lessons → rag → plan → task
        idx_skills = result.index("[skills]")
        idx_lessons = result.index("[lessons]")
        idx_rag = result.index("[rag]")
        idx_plan = result.index("[任務分析計劃]")
        idx_task = result.index("question")

        assert idx_skills < idx_lessons < idx_rag < idx_plan < idx_task

    def test_task_only(self):
        """With only a task, output is just the task."""
        ca = ContextAssembler()
        assert ca.assemble(task="hello") == "hello"

    def test_optional_parts_skipped(self):
        """None/empty parts are excluded."""
        ca = ContextAssembler()
        result = ca.assemble(task="q", skills_block=None, lessons="")
        assert result == "q"


# ═══════════════════════════════════════════════════════════════════
#  Tests — HallucinationGuard
# ═══════════════════════════════════════════════════════════════════

class TestHallucinationGuard:

    def test_no_claims_high_confidence(self):
        """Answer with no factual claims → high confidence."""
        guard = HallucinationGuard()
        result = guard.verify("OK", "task", [])

        assert result.verified is True
        assert result.confidence >= 0.9

    def test_heuristic_extracts_claims(self):
        """Heuristic extraction finds sentences with numbers."""
        claims = HallucinationGuard._extract_claims_heuristic(
            "The Earth is approximately 1.5 million km from the Sun. "
            "Water boils at 100 degrees Celsius at sea level."
        )
        assert len(claims) >= 1
        assert any("1.5" in c for c in claims)

    def test_trace_verification(self):
        """Claims found in trace observations are verified."""
        result = HallucinationGuard._check_against_trace(
            "The answer is 42",
            ["Tool returned: The answer is 42"],
        )
        assert result is True

    def test_trace_verification_miss(self):
        """Claims not in trace are not verified."""
        result = HallucinationGuard._check_against_trace(
            "Earth is the third planet",
            ["Tool returned: weather is sunny"],
        )
        assert result is False

    def test_llm_claim_extraction_fallback(self):
        """When LLM extraction fails, heuristic is used."""
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=RuntimeError("fail"))
        guard = HallucinationGuard(llm=llm)

        # Should not raise, falls back to heuristic
        result = guard.verify(
            "地球距離太陽約 1.5 億公里。",
            "task", [],
        )
        assert isinstance(result, VerificationResult)
