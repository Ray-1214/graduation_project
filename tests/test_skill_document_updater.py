"""
Tests for SkillDocumentUpdater.

Covers:
  - Trigger conditions (failure, discovery, divergence, none)
  - Template-based updates (caveats, strategy, version history)
  - LLM-based updates (mock) + fallback on short/exception
  - Backup creation
  - Quality gate
  - Version increment
  - Edge cases (no document_path, missing file)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.episodic_log import EpisodicTrace, TraceStep
from skill_graph.skill_document_updater import SkillDocumentUpdater, UpdateResult
from skill_graph.skill_node import SkillNode


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

_SAMPLE_DOC = """\
# 搜尋技能

## 適用場景
- 問答
- 知識查詢

## 前置條件
無特殊前置條件。

## 策略步驟
1. 使用 web_search 搜尋關鍵字
2. 閱讀搜尋結果
3. 提取關鍵資訊
4. 組織成摘要

## 終止條件
找到明確答案後結束。

## 注意事項
*Phase 3 反思機制將自動填充。*

## 版本歷史
| 版本 | 日期 | 備註 |
|------|------|------|
| v1 | 2026-01-01 | 初始版本 |
"""


def _skill(**kw) -> SkillNode:
    defaults = dict(
        skill_id="sk-test-001",
        name="搜尋技能",
        policy="search → read → summarize",
        termination="找到明確答案後結束",
        initiation_set=["問答", "知識查詢"],
        version=1,
        document_path="skills/learned/sk-test-001.md",
    )
    defaults.update(kw)
    return SkillNode(**defaults)


def _trace(
    tid: str = "trace-001",
    actions: list[str] | None = None,
    success: bool = True,
) -> EpisodicTrace:
    if actions is None:
        actions = ["web_search", "閱讀結果", "提取資訊", "寫摘要"]
    steps = [
        TraceStep(
            state=f"s{i}", action=a,
            outcome=f"outcome_{i}", timestamp=time.time(),
        )
        for i, a in enumerate(actions)
    ]
    return EpisodicTrace(
        task_id=tid, task_description="test task",
        steps=steps, strategy="react",
        success=success, score=0.8 if success else 0.2,
        total_time=2.0,
    )


def _write_doc(tmp_path: Path, content: str = _SAMPLE_DOC) -> Path:
    """Create the skill document on disk."""
    doc_dir = tmp_path / "skills" / "learned"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / "sk-test-001.md"
    doc_path.write_text(content, encoding="utf-8")
    return doc_path


# ═══════════════════════════════════════════════════════════════════
#  Trigger Conditions
# ═══════════════════════════════════════════════════════════════════

class TestTriggers:

    def test_failure_triggers_update(self, tmp_path):
        """Task failure → trigger = failure_caveat."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        result = updater.update(
            _skill(), _trace(success=False),
            "搜尋結果不相關，答案錯誤", success=False,
        )
        assert result.updated is True
        assert result.trigger == "failure_caveat"

    def test_discovery_triggers_update(self, tmp_path):
        """Success + reflexion mentions new discovery → trigger."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        result = updater.update(
            _skill(), _trace(),
            "發現可以先用 wiki 搜尋再用 web_search，效果更好",
            success=True,
        )
        assert result.updated is True
        assert result.trigger == "new_discovery"

    def test_divergence_triggers_update(self, tmp_path):
        """Success + trace very different from doc → trigger."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(
            llm=None, root=tmp_path, divergence_threshold=0.3,
        )
        trace = _trace(actions=["計算", "推理", "驗證", "輸出"])
        result = updater.update(
            _skill(), trace,
            "任務完成了", success=True,
        )
        assert result.updated is True
        assert result.trigger == "strategy_divergence"

    def test_no_trigger_on_routine_success(self, tmp_path):
        """Success + no discovery + similar trace → no update."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(
            llm=None, root=tmp_path, divergence_threshold=0.9,
        )
        result = updater.update(
            _skill(), _trace(),
            "任務順利完成", success=True,
        )
        assert result.updated is False
        assert result.trigger is None

    def test_no_document_path(self, tmp_path):
        """No document_path → skip update."""
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        skill = _skill(document_path=None)
        result = updater.update(skill, _trace(), "test", success=True)
        assert result.updated is False
        assert "No document_path" in result.reason

    def test_missing_file(self, tmp_path):
        """document_path set but file missing → skip update."""
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        result = updater.update(
            _skill(), _trace(), "test", success=False,
        )
        assert result.updated is False
        assert "not found" in result.reason


# ═══════════════════════════════════════════════════════════════════
#  Template Updates
# ═══════════════════════════════════════════════════════════════════

class TestTemplateUpdates:

    def test_failure_adds_caveat(self, tmp_path):
        """Failure → 注意事項 section gets a warning."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        updater.update(
            _skill(), _trace("fail-001", success=False),
            "搜尋關鍵字選擇不當", success=False,
        )
        doc_path = tmp_path / "skills" / "learned" / "sk-test-001.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "⚠️" in content
        assert "fail-001" in content
        assert "搜尋關鍵字選擇不當" in content

    def test_discovery_adds_strategy(self, tmp_path):
        """Discovery → 策略步驟 section gets new info."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        updater.update(
            _skill(), _trace("disc-001"),
            "發現先查 wiki 比直接搜尋更準確", success=True,
        )
        doc_path = tmp_path / "skills" / "learned" / "sk-test-001.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "新增" in content
        assert "disc-001" in content

    def test_divergence_adds_alternate_path(self, tmp_path):
        """Divergence → alternate path appended to strategy."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(
            llm=None, root=tmp_path, divergence_threshold=0.3,
        )
        trace = _trace("div-001", actions=["計算", "推理", "驗證"])
        updater.update(
            _skill(), trace,
            "任務完成了", success=True,
        )
        doc_path = tmp_path / "skills" / "learned" / "sk-test-001.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "替代路徑" in content
        assert "div-001" in content

    def test_version_history_appended(self, tmp_path):
        """Every update appends to version history."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        updater.update(
            _skill(), _trace(success=False),
            "失敗了", success=False,
        )
        doc_path = tmp_path / "skills" / "learned" / "sk-test-001.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "v2" in content
        assert "failure_caveat" in content


# ═══════════════════════════════════════════════════════════════════
#  Version & Backup
# ═══════════════════════════════════════════════════════════════════

class TestVersionAndBackup:

    def test_version_incremented(self, tmp_path):
        """skill.version is bumped from 1 to 2."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        skill = _skill(version=1)
        result = updater.update(
            skill, _trace(success=False), "fail", success=False,
        )
        assert skill.version == 2
        assert result.new_version == 2

    def test_backup_created(self, tmp_path):
        """Backup file {skill_id}_v{old}.md.bak is created."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        result = updater.update(
            _skill(), _trace(success=False), "fail", success=False,
        )
        assert result.backup_path is not None
        backup = Path(result.backup_path)
        assert backup.exists()
        assert "v1.md.bak" in backup.name

    def test_backup_preserves_original(self, tmp_path):
        """Backup content matches original document."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        result = updater.update(
            _skill(), _trace(success=False), "fail", success=False,
        )
        backup_content = Path(result.backup_path).read_text(encoding="utf-8")
        assert backup_content == _SAMPLE_DOC

    def test_multiple_updates_increment(self, tmp_path):
        """Two updates → version goes from 1 to 3."""
        _write_doc(tmp_path)
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        skill = _skill(version=1)

        updater.update(skill, _trace(success=False), "first fail", success=False)
        assert skill.version == 2

        updater.update(skill, _trace(success=False), "second fail", success=False)
        assert skill.version == 3


# ═══════════════════════════════════════════════════════════════════
#  LLM Mode
# ═══════════════════════════════════════════════════════════════════

class TestLLMMode:

    def test_llm_update_used(self, tmp_path):
        """When LLM returns valid update → used as new content."""
        _write_doc(tmp_path)
        mock_llm = MagicMock()
        # Return a valid updated document
        updated_doc = _SAMPLE_DOC.replace(
            "*Phase 3 反思機制將自動填充。*",
            "- ⚠️ 注意確認搜尋結果的可靠性\n- 策略步驟的改進建議",
        )
        mock_llm.generate.return_value = updated_doc

        updater = SkillDocumentUpdater(llm=mock_llm, root=tmp_path)
        result = updater.update(
            _skill(), _trace(success=False), "搜尋失敗", success=False,
        )
        assert result.updated is True
        mock_llm.generate.assert_called_once()
        doc_content = (
            tmp_path / "skills" / "learned" / "sk-test-001.md"
        ).read_text(encoding="utf-8")
        assert "注意確認搜尋結果的可靠性" in doc_content

    def test_llm_short_output_falls_back(self, tmp_path):
        """LLM returns too-short output → template fallback."""
        _write_doc(tmp_path)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "太短了"

        updater = SkillDocumentUpdater(llm=mock_llm, root=tmp_path)
        result = updater.update(
            _skill(), _trace(success=False), "失敗", success=False,
        )
        assert result.updated is True
        # Template fallback: should have ⚠️ marker
        doc_content = (
            tmp_path / "skills" / "learned" / "sk-test-001.md"
        ).read_text(encoding="utf-8")
        assert "⚠️" in doc_content

    def test_llm_exception_falls_back(self, tmp_path):
        """LLM raises exception → template fallback."""
        _write_doc(tmp_path)
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("OOM")

        updater = SkillDocumentUpdater(llm=mock_llm, root=tmp_path)
        result = updater.update(
            _skill(), _trace(success=False), "失敗", success=False,
        )
        assert result.updated is True


# ═══════════════════════════════════════════════════════════════════
#  Quality Gate
# ═══════════════════════════════════════════════════════════════════

class TestQualityGate:

    def test_empty_update_rejected(self, tmp_path):
        """LLM returns empty string → quality gate rejects."""
        _write_doc(tmp_path)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ""

        updater = SkillDocumentUpdater(llm=mock_llm, root=tmp_path)
        # Template fallback will still produce valid content, so this
        # tests the LLM path specifically
        result = updater.update(
            _skill(), _trace(success=False), "fail", success=False,
        )
        # Template fallback should still work
        assert result.updated is True

    def test_quality_gate_preserves_original_on_bad_llm(self, tmp_path):
        """When LLM output is garbage and template is disabled somehow,
        quality gate prevents data loss."""
        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        # Test the static method directly
        assert not updater._passes_quality_gate("", _SAMPLE_DOC)
        assert not updater._passes_quality_gate("short", _SAMPLE_DOC)
        assert updater._passes_quality_gate(_SAMPLE_DOC, _SAMPLE_DOC)


# ═══════════════════════════════════════════════════════════════════
#  Repr
# ═══════════════════════════════════════════════════════════════════

class TestRepr:

    def test_repr_template_mode(self):
        updater = SkillDocumentUpdater(llm=None)
        assert "template" in repr(updater)

    def test_repr_llm_mode(self):
        updater = SkillDocumentUpdater(llm=MagicMock())
        assert "LLM" in repr(updater)
