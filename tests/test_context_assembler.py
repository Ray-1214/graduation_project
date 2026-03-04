"""
Tests for the slot-based ContextAssembler.

Covers:
  - Slot construction and budget defaults
  - Recency-weighted history compression (3 tiers)
  - Knowledge formatting with source separation
  - Compression cascade (overflow handling)
  - Token estimation
  - Budget reporting
  - Legacy backward compatibility
  - AssembledContext fields
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning.context_assembler import (
    AssembledContext,
    ContextAssembler,
    ContextSlot,
    ReActStep,
    estimate_tokens,
    _CHARS_PER_TOKEN,
)


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_steps(n: int) -> list[ReActStep]:
    """Create n ReAct steps for testing."""
    return [
        ReActStep(
            thought=f"Thinking about step {i}",
            action=f"web_search: query {i}",
            observation=f"Result for query {i} with details",
            step_num=i,
        )
        for i in range(1, n + 1)
    ]


def _make_knowledge_entry(content, source="web", confidence=0.8):
    """Create a mock KnowledgeEntry."""
    entry = MagicMock()
    entry.content = content
    entry.source = source
    entry.confidence = confidence
    entry.query = f"query for {content[:20]}"
    return entry


# ═══════════════════════════════════════════════════════════════════
#  Tests — Token Estimation
# ═══════════════════════════════════════════════════════════════════

class TestTokenEstimation:

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        assert estimate_tokens("hello") >= 1

    def test_proportional_to_length(self):
        short = estimate_tokens("abc")
        long_ = estimate_tokens("a" * 100)
        assert long_ > short

    def test_chars_per_token_ratio(self):
        text = "a" * 400
        assert estimate_tokens(text) == 400 // _CHARS_PER_TOKEN


# ═══════════════════════════════════════════════════════════════════
#  Tests — ContextSlot
# ═══════════════════════════════════════════════════════════════════

class TestContextSlot:

    def test_defaults(self):
        slot = ContextSlot(name="test")
        assert slot.priority == 5
        assert slot.max_tokens == 500
        assert slot.compressible is True
        assert slot.compressed is False
        assert slot.content == ""

    def test_custom_values(self):
        slot = ContextSlot(
            name="system", priority=0, max_tokens=300,
            content="hello", compressible=False,
        )
        assert slot.name == "system"
        assert slot.compressible is False


# ═══════════════════════════════════════════════════════════════════
#  Tests — Recency-Weighted History
# ═══════════════════════════════════════════════════════════════════

class TestRecencyWeightedHistory:

    def test_empty_history(self):
        ca = ContextAssembler()
        result = ca._compress_history([], 1500)
        assert result == ""

    def test_single_step_full(self):
        """Single step is kept verbatim (tier 1)."""
        ca = ContextAssembler()
        steps = _make_steps(1)
        result = ca._compress_history(steps, 1500)
        assert "Thought: Thinking about step 1" in result
        assert "Action: web_search: query 1" in result
        assert "Observation: Result for query 1" in result

    def test_two_steps_both_full(self):
        """Last 2 steps kept verbatim."""
        ca = ContextAssembler()
        steps = _make_steps(2)
        result = ca._compress_history(steps, 1500)
        assert "Thought: Thinking about step 1" in result
        assert "Thought: Thinking about step 2" in result

    def test_four_steps_mixed_tiers(self):
        """4 steps: steps 1-2 condensed (tier 2), steps 3-4 full."""
        ca = ContextAssembler()
        steps = _make_steps(4)
        result = ca._compress_history(steps, 5000)

        # Last 2 steps (3,4) should be fully present
        assert "Action: web_search: query 4" in result
        # Middle steps should be condensed (have | separators)
        # Step 1 and 2 are in mid tier
        assert "…" in result  # condensed marker

    def test_eight_steps_three_tiers(self):
        """8 steps: 1-3 summarised, 4-6 condensed, 7-8 full."""
        ca = ContextAssembler()
        steps = _make_steps(8)
        result = ca._compress_history(steps, 10000)

        # Tier 3: old steps summary
        assert "步驟 1-3 摘要" in result
        # Tier 1: last 2 steps full
        assert "Thought: Thinking about step 8" in result

    def test_history_truncated_when_over_budget(self):
        """History exceeding budget is trimmed by removing oldest parts."""
        ca = ContextAssembler()
        steps = _make_steps(10)
        result = ca._compress_history(steps, 100)  # very tight budget
        tokens = estimate_tokens(result)
        # Should be trimmed (may not perfectly fit but should try)
        assert tokens <= 200  # generous margin for heuristic


# ═══════════════════════════════════════════════════════════════════
#  Tests — Knowledge Formatting
# ═══════════════════════════════════════════════════════════════════

class TestKnowledgeFormatting:

    def test_empty_knowledge(self):
        ca = ContextAssembler()
        result = ca._format_knowledge([], None, 500)
        assert result == ""

    def test_external_entries_formatted(self):
        """Web-sourced entries go under external section."""
        ca = ContextAssembler()
        entries = [_make_knowledge_entry("Python is great", "web", 0.9)]
        result = ca._format_knowledge(entries, None, 500)
        assert "外部來源" in result
        assert "Python is great" in result

    def test_reflexion_entries_separated(self):
        """Reflexion-sourced entries go under separate section."""
        ca = ContextAssembler()
        entries = [
            _make_knowledge_entry("fact from web", "web", 0.9),
            _make_knowledge_entry("insight from reflexion", "reflexion", 0.6),
        ]
        result = ca._format_knowledge(entries, None, 2000)
        assert "外部來源" in result
        assert "Agent 經驗" in result
        assert "可信度較低" in result

    def test_legacy_rag_context(self):
        """When no entries but rag_context given, use it directly."""
        ca = ContextAssembler()
        result = ca._format_knowledge([], "[rag data]", 500)
        assert "[rag data]" in result

    def test_budget_respected(self):
        """Knowledge entries exceeding budget are truncated."""
        ca = ContextAssembler()
        entries = [
            _make_knowledge_entry("x" * 500, "web", 0.9),
            _make_knowledge_entry("y" * 500, "web", 0.8),
            _make_knowledge_entry("z" * 500, "web", 0.7),
        ]
        result = ca._format_knowledge(entries, None, 50)  # very tight
        # Should not include all 3
        assert result.count("y" * 100) <= 1


# ═══════════════════════════════════════════════════════════════════
#  Tests — Lessons Formatting
# ═══════════════════════════════════════════════════════════════════

class TestLessonsFormatting:

    def test_empty(self):
        assert ContextAssembler._format_lessons([]) == ""

    def test_formatted(self):
        result = ContextAssembler._format_lessons(["lesson 1", "lesson 2"])
        assert "歷史教訓" in result
        assert "- lesson 1" in result
        assert "- lesson 2" in result


# ═══════════════════════════════════════════════════════════════════
#  Tests — Full assemble()
# ═══════════════════════════════════════════════════════════════════

class TestAssemble:

    def test_basic_assembly(self):
        """Basic assembly returns AssembledContext."""
        ca = ContextAssembler()
        ctx = ca.assemble(task="What is 2+2?")

        assert isinstance(ctx, AssembledContext)
        assert "What is 2+2?" in ctx.prompt
        assert ctx.total_tokens > 0

    def test_all_slots_present(self):
        """All configured slots appear in the report."""
        ca = ContextAssembler()
        ctx = ca.assemble(
            task="task",
            system_prompt="system",
            history=_make_steps(2),
            knowledge=[_make_knowledge_entry("fact", "web")],
            lessons=["lesson"],
            skill_docs=["skill doc"],
            tools=["tool1"],
        )

        report = ctx.slot_report
        assert "system" in report
        assert "task" in report
        assert "knowledge" in report
        assert "lessons" in report
        assert "skills" in report
        assert "history" in report
        assert "tools" in report
        assert "reserve" in report

    def test_report_has_budget_and_used(self):
        """Each slot in report has budget, used, compressed keys."""
        ca = ContextAssembler()
        ctx = ca.assemble(task="hello")

        for name, info in ctx.slot_report.items():
            assert "budget" in info
            assert "used" in info
            assert "compressed" in info

    def test_system_prompt_included(self):
        ca = ContextAssembler()
        ctx = ca.assemble(task="q", system_prompt="You are helpful")
        assert "You are helpful" in ctx.prompt

    def test_tools_included(self):
        ca = ContextAssembler()
        ctx = ca.assemble(task="q", tools=["web_search: search the web"])
        assert "web_search" in ctx.prompt

    def test_thinking_plan_appended_to_task(self):
        """Thinking plan is embedded in the task slot."""
        ca = ContextAssembler()
        ctx = ca.assemble(task="my task", thinking_plan="step 1, step 2")
        assert "[任務分析計劃]" in ctx.prompt
        assert "step 1, step 2" in ctx.prompt

    def test_history_included(self):
        """History steps make it into the prompt."""
        ca = ContextAssembler()
        steps = _make_steps(2)
        ctx = ca.assemble(task="task", history=steps)
        assert "web_search" in ctx.prompt


# ═══════════════════════════════════════════════════════════════════
#  Tests — Compression
# ═══════════════════════════════════════════════════════════════════

class TestCompression:

    def test_no_overflow_no_compression(self):
        """Under budget → no overflow, no compressed slots."""
        ca = ContextAssembler(max_total_tokens=10000)
        ctx = ca.assemble(task="short task")
        assert ctx.overflow is False
        assert ctx.compressed_slots == []

    def test_overflow_triggers_compression(self):
        """Over budget → overflow=True, slots compressed."""
        ca = ContextAssembler(max_total_tokens=50)  # very small
        ctx = ca.assemble(
            task="a" * 200,
            history=_make_steps(5),
            lessons=["big lesson " * 20],
        )
        # Should detect overflow and attempt compression
        assert ctx.overflow is True

    def test_slot_compress_with_llm(self):
        """LLM compression produces shorter content."""
        llm = MagicMock()
        llm.generate = MagicMock(return_value="compressed summary")
        ca = ContextAssembler(llm=llm)

        slot = ContextSlot(
            name="test", content="x" * 2000,
            max_tokens=50, compressible=True,
        )
        ca._compress_slot(slot)
        assert slot.compressed is True
        assert llm.generate.called

    def test_slot_compress_fallback_truncate(self):
        """When LLM unavailable, truncation is used."""
        ca = ContextAssembler(llm=None)

        slot = ContextSlot(
            name="test", content="x" * 2000,
            max_tokens=50, compressible=True,
        )
        ca._compress_slot(slot)
        assert slot.compressed is True
        assert len(slot.content) < 2000

    def test_truncation_keeps_recent(self):
        """Truncation keeps the end (most recent) of the text."""
        ca = ContextAssembler()
        content = "\n".join([f"line {i}" for i in range(100)])
        result = ca._truncate(content, 20)  # very tight
        # Should keep the tail
        assert "壓縮" in result  # truncation marker
        assert "line 99" in result  # recent content kept


# ═══════════════════════════════════════════════════════════════════
#  Tests — Budget Report
# ═══════════════════════════════════════════════════════════════════

class TestBudgetReport:

    def test_get_budget_report(self):
        """Budget report matches last assembly."""
        ca = ContextAssembler()
        ca.assemble(task="hello")
        report = ca.get_budget_report()

        assert isinstance(report, dict)
        assert "task" in report
        assert report["task"]["budget"] == 200

    def test_report_empty_before_assembly(self):
        ca = ContextAssembler()
        assert ca.get_budget_report() == {}


# ═══════════════════════════════════════════════════════════════════
#  Tests — Custom Slot Budgets
# ═══════════════════════════════════════════════════════════════════

class TestCustomBudgets:

    def test_custom_budget_override(self):
        """Custom budgets override defaults."""
        ca = ContextAssembler(
            slot_budgets={"task": {"priority": 0, "max_tokens": 999,
                                   "compressible": False}}
        )
        ctx = ca.assemble(task="hello")
        assert ctx.slot_report["task"]["budget"] == 999

    def test_max_total_tokens(self):
        """Custom max_total_tokens is respected."""
        ca = ContextAssembler(max_total_tokens=2000)
        assert ca.max_total_tokens == 2000


# ═══════════════════════════════════════════════════════════════════
#  Tests — AssembledContext
# ═══════════════════════════════════════════════════════════════════

class TestAssembledContext:

    def test_defaults(self):
        ctx = AssembledContext()
        assert ctx.prompt == ""
        assert ctx.total_tokens == 0
        assert ctx.overflow is False
        assert ctx.compressed_slots == []
        assert ctx.slot_report == {}
