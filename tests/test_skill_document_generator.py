"""
Tests for SkillDocumentGenerator.

Covers:
  - Template-only mode: generates valid Markdown with all sections
  - LLM mode: mock-based verification of prompt + output
  - LLM fallback: short/failed output → template used
  - Tool action detection in traces
  - KnowledgeEntry inclusion
  - File persistence + skill.document_path update
  - Edge cases: empty traces, no initiation set
  - Version history format
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.episodic_log import EpisodicTrace, TraceStep
from rag.knowledge_store import KnowledgeEntry
from skill_graph.skill_document_generator import SkillDocumentGenerator
from skill_graph.skill_node import SkillNode


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_skill(**kw) -> SkillNode:
    defaults = dict(
        skill_id="sk-test-001",
        name="搜尋並總結",
        policy="search → read → summarize",
        termination="找到明確答案後結束",
        initiation_set=["問答", "知識查詢"],
        tags=["abstracted"],
    )
    defaults.update(kw)
    return SkillNode(**defaults)


def _make_trace(
    task_id: str = "trace-001",
    actions: list[str] | None = None,
    score: float = 0.8,
) -> EpisodicTrace:
    if actions is None:
        actions = [
            'Action[web_search]("Python GIL")',
            "閱讀搜尋結果",
            "提取關鍵概念",
            "組織成摘要",
        ]
    steps = [
        TraceStep(
            state=f"s{i}",
            action=a,
            outcome=f"outcome_{i}",
            timestamp=time.time(),
        )
        for i, a in enumerate(actions)
    ]
    return EpisodicTrace(
        task_id=task_id,
        task_description="test task",
        steps=steps,
        strategy="react",
        success=True,
        score=score,
        total_time=2.0,
    )


def _make_knowledge_entry(**kw) -> KnowledgeEntry:
    defaults = dict(
        query="What is GIL?",
        content="GIL stands for Global Interpreter Lock...",
        source="web",
        confidence=0.8,
        url="https://example.com/gil",
    )
    defaults.update(kw)
    return KnowledgeEntry(**defaults)


# ═══════════════════════════════════════════════════════════════════
#  Template Mode (no LLM)
# ═══════════════════════════════════════════════════════════════════

class TestTemplateMode:

    def test_generates_all_sections(self, tmp_path):
        """Output contains all 7 required sections."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill()
        traces = [_make_trace()]

        path = gen.generate(skill, traces)

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "# 搜尋並總結" in content
        assert "## 適用場景" in content
        assert "## 前置條件" in content
        assert "## 策略步驟" in content
        assert "## 終止條件" in content
        assert "## 注意事項" in content
        assert "## 版本歷史" in content

    def test_applicability_from_initiation_set(self, tmp_path):
        """適用場景 lists initiation_set items."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill(initiation_set=["數學", "推理"])
        path = gen.generate(skill, [_make_trace()])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "- 數學" in content
        assert "- 推理" in content

    def test_termination_from_skill(self, tmp_path):
        """終止條件 uses the skill's termination predicate."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill(termination="所有步驟完成後停止")
        path = gen.generate(skill, [_make_trace()])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "所有步驟完成後停止" in content

    def test_tool_action_detected(self, tmp_path):
        """Tool actions (Action[...](args)) are formatted specially."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill()
        trace = _make_trace(actions=[
            'Action[web_search]("query")',
            "normal step",
        ])
        path = gen.generate(skill, [trace])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "Action[web_search]" in content

    def test_strategy_from_policy_when_no_traces(self, tmp_path):
        """When no traces given, strategy steps come from policy string."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill(policy="step_a → step_b → step_c")
        path = gen.generate(skill, [])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "step_a" in content
        assert "step_b" in content
        assert "step_c" in content

    def test_best_trace_used(self, tmp_path):
        """With multiple traces, the highest-scoring one is used."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill()
        low = _make_trace("low", ["bad_step"], score=0.2)
        high = _make_trace("high", ["good_step"], score=0.9)
        path = gen.generate(skill, [low, high])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "high" in content  # best trace_id


# ═══════════════════════════════════════════════════════════════════
#  File Persistence
# ═══════════════════════════════════════════════════════════════════

class TestPersistence:

    def test_file_created(self, tmp_path):
        """Document file is physically written to disk."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill()
        path = gen.generate(skill, [_make_trace()])

        assert (tmp_path / path).exists()
        assert path.endswith(".md")

    def test_document_path_updated(self, tmp_path):
        """skill.document_path is set to the relative path."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill()
        path = gen.generate(skill, [_make_trace()])

        assert skill.document_path == path
        assert skill.document_path.startswith("skills/learned/")

    def test_directory_created(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        gen = SkillDocumentGenerator(
            llm=None, output_dir="deep/nested/dir", root=tmp_path,
        )
        skill = _make_skill()
        gen.generate(skill, [_make_trace()])

        assert (tmp_path / "deep" / "nested" / "dir").is_dir()


# ═══════════════════════════════════════════════════════════════════
#  Knowledge Entry Inclusion
# ═══════════════════════════════════════════════════════════════════

class TestKnowledgeInclusion:

    def test_knowledge_section_present(self, tmp_path):
        """When knowledge entries provided, 背景知識 section appears."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill()
        entry = _make_knowledge_entry()
        path = gen.generate(skill, [_make_trace()], [entry])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "## 背景知識" in content
        assert "GIL" in content

    def test_knowledge_section_absent_without_entries(self, tmp_path):
        """Without knowledge entries, no 背景知識 section."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill()
        path = gen.generate(skill, [_make_trace()])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "## 背景知識" not in content

    def test_knowledge_url_included(self, tmp_path):
        """Entry URL is shown when present."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        entry = _make_knowledge_entry(url="https://docs.python.org")
        path = gen.generate(_make_skill(), [_make_trace()], [entry])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "https://docs.python.org" in content


# ═══════════════════════════════════════════════════════════════════
#  LLM Mode
# ═══════════════════════════════════════════════════════════════════

class TestLLMMode:

    def test_llm_generate_called(self, tmp_path):
        """LLM.generate is called with a prompt containing trace info."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "1. 首先在網路上搜尋相關資訊\n"
            "2. 閱讀並理解搜尋結果\n"
            "3. 提取關鍵資訊並組織\n"
            "4. 生成簡明的摘要回覆使用者\n"
        )

        gen = SkillDocumentGenerator(llm=mock_llm, root=tmp_path)
        gen.generate(_make_skill(), [_make_trace()])

        mock_llm.generate.assert_called_once()
        prompt_arg = mock_llm.generate.call_args[0][0]
        assert "搜尋並總結" in prompt_arg  # skill name in prompt
        assert "trace" in prompt_arg.lower()

    def test_llm_output_included(self, tmp_path):
        """When LLM output is valid, it appears in the document."""
        mock_llm = MagicMock()
        llm_text = (
            "1. 使用 Action[web_search] 搜尋目標關鍵字\n"
            "2. 分析返回的搜尋結果，識別最相關的條目\n"
            "3. 提取核心資訊並組織成結構化摘要\n"
        )
        mock_llm.generate.return_value = llm_text

        gen = SkillDocumentGenerator(llm=mock_llm, root=tmp_path)
        path = gen.generate(_make_skill(), [_make_trace()])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "Action[web_search]" in content

    def test_llm_fallback_on_short_output(self, tmp_path):
        """If LLM returns too little text, template fallback is used."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "太短"  # < 80 chars

        gen = SkillDocumentGenerator(llm=mock_llm, root=tmp_path)
        path = gen.generate(_make_skill(), [_make_trace()])

        content = (tmp_path / path).read_text(encoding="utf-8")
        # Template fallback includes trace ID
        assert "trace-001" in content

    def test_llm_fallback_on_exception(self, tmp_path):
        """If LLM raises an exception, template fallback is used gracefully."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("GPU OOM")

        gen = SkillDocumentGenerator(llm=mock_llm, root=tmp_path)
        path = gen.generate(_make_skill(), [_make_trace()])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "## 策略步驟" in content  # still has the section
        assert "trace-001" in content    # from template


# ═══════════════════════════════════════════════════════════════════
#  Edge Cases & Version History
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_initiation_set(self, tmp_path):
        """Empty initiation set shows placeholder."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill(initiation_set=[])
        path = gen.generate(skill, [_make_trace()])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "尚未定義" in content

    def test_evolved_skill_shows_parent(self, tmp_path):
        """Evolved skill mentions parent_id in prerequisites."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill(parent_id="sk-parent-001", version=2)
        path = gen.generate(skill, [_make_trace()])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "sk-parent-001" in content

    def test_version_history_format(self, tmp_path):
        """Version history includes version number, date, trace IDs."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _make_skill()
        trace = _make_trace(task_id="tr-abc-123")
        path = gen.generate(skill, [trace])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "v1" in content
        assert "tr-abc-123" in content

    def test_caveats_initially_empty(self, tmp_path):
        """注意事項 section is placeholder text."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        path = gen.generate(_make_skill(), [_make_trace()])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "Phase 3" in content

    def test_repr(self):
        """__repr__ shows mode."""
        gen = SkillDocumentGenerator(llm=None)
        assert "template" in repr(gen)

        gen2 = SkillDocumentGenerator(llm=MagicMock())
        assert "LLM" in repr(gen2)
