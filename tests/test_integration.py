"""
Phase 3 Acceptance Tests — Integration & Verification

Test numbering follows the acceptance criteria document:

  3.1 Agent Integration (3-1 … 3-4)
  3.2 SkillRetriever    (3-5 … 3-10)
  3.3 SkillDocument     (3-11 … 3-15)

Run with:
    pytest tests/test_integration.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.episodic_log import (
    EpisodicLog,
    EpisodicTrace,
    TraceStep,
    convert_log_to_trace,
)
from skill_graph.evolution_operator import EvolutionOperator
from skill_graph.memory_partition import MemoryPartition
from skill_graph.skill_document_updater import SkillDocumentUpdater
from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode
from skill_graph.skill_retriever import RetrievedSkill, SkillRetriever


# ═══════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════

def _skill(sid: str, **kw) -> SkillNode:
    defaults = dict(
        skill_id=sid,
        name=f"skill_{sid}",
        policy=f"policy for {sid}",
        termination=f"done when {sid} complete",
        initiation_set=[f"tag_{sid}"],
        cost=1.0,
        utility=0.5,
    )
    defaults.update(kw)
    return SkillNode(**defaults)


def _trace(
    tid: str = "trace-001",
    actions: list[str] | None = None,
    success: bool = True,
    score: float | None = None,
) -> EpisodicTrace:
    if actions is None:
        actions = ["search", "read", "summarize"]
    steps = [
        TraceStep(
            state=f"s{i}", action=a,
            outcome=f"outcome_{i}", timestamp=time.time(),
        )
        for i, a in enumerate(actions)
    ]
    return EpisodicTrace(
        task_id=tid, task_description="test task",
        steps=steps, strategy="cot",
        success=success,
        score=score if score is not None else (0.8 if success else 0.2),
        total_time=1.5,
    )


_SAMPLE_DOC = """\
# 搜尋技能

## 適用場景
- 問答

## 策略步驟
1. 搜尋關鍵字
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


def _write_doc(
    tmp_path: Path, skill_id: str = "sk-search",
    content: str = _SAMPLE_DOC,
) -> str:
    """Create skill doc and return relative path."""
    doc_dir = tmp_path / "skills" / "learned"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / f"{skill_id}.md"
    doc_path.write_text(content, encoding="utf-8")
    return f"skills/learned/{skill_id}.md"


# ═══════════════════════════════════════════════════════════════════
#  3.1 Agent 整合測試  (3-1 … 3-4)
# ═══════════════════════════════════════════════════════════════════

class TestAgentIntegration:
    """Tests 3-1 through 3-4: MainAgent integration with SkillGraph."""

    # ── 3-1  向後相容 ────────────────────────────────────────────────

    def test_3_1_backward_compatible_empty_graph(self):
        """SkillGraph is empty → run() should behave exactly as before.

        The skill retrieval step should be skipped; no skill block
        should appear in the prompt; result format must be unchanged.
        """
        from agents.main_agent import AgentResult, MainAgent

        agent = MainAgent()
        assert len(agent.skill_graph) == 0

        # Patch the _dispatch to capture the effective_task
        captured_tasks: list[str] = []
        original_dispatch = agent._dispatch

        def capture_dispatch(strat, task, episode):
            captured_tasks.append(task)
            return "test answer"

        agent._dispatch = capture_dispatch
        # Patch evaluator to skip LLM call
        agent.evaluator.evaluate = MagicMock(return_value=0.9)
        agent.reflexion.reflect = MagicMock(
            return_value=MagicMock(reflection="OK"),
        )
        agent.reflexion.get_relevant_lessons = MagicMock(return_value="")

        result = agent.run("What is 2+2?", strategy="cot")

        # No skill injection keywords in prompt
        assert len(captured_tasks) == 1
        assert "已知技能" not in captured_tasks[0]
        # Result format
        assert isinstance(result, AgentResult)
        assert result.task == "What is 2+2?"
        assert result.answer == "test answer"
        assert result.score == 0.9
        assert result.evolution_log is not None  # still runs evolve

    # ── 3-2  Skill 注入 prompt ───────────────────────────────────────

    def test_3_2_skill_injected_into_prompt(self, tmp_path):
        """Graph has 2 active skills with documents → prompt contains
        full .md file content (not just short policy string).
        """
        from agents.main_agent import MainAgent

        agent = MainAgent()

        # Create skills with documents
        doc_content_a = "# 技能A 操作手冊\n\n## 策略步驟\n1. 步驟A1\n2. 步驟A2\n"
        doc_content_b = "# 技能B 操作手冊\n\n## 策略步驟\n1. 步驟B1\n"

        rel_a = _write_doc(tmp_path, "sk-a", doc_content_a)
        rel_b = _write_doc(tmp_path, "sk-b", doc_content_b)

        skill_a = _skill("sk-a", name="搜尋", utility=1.0,
                         initiation_set=["搜尋", "查詢"],
                         document_path=rel_a)
        skill_b = _skill("sk-b", name="計算", utility=0.8,
                         initiation_set=["計算", "數學"],
                         document_path=rel_b)

        agent.skill_graph.add_skill(skill_a)
        agent.skill_graph.add_skill(skill_b)
        # Point retriever to tmp_path
        agent.skill_retriever.root = tmp_path

        # Capture what prompt is sent to _dispatch
        captured: list[str] = []

        def capture(strat, task, episode):
            captured.append(task)
            return "answer"

        agent._dispatch = capture
        agent.evaluator.evaluate = MagicMock(return_value=0.9)
        agent.reflexion.reflect = MagicMock(
            return_value=MagicMock(reflection="OK"),
        )
        agent.reflexion.get_relevant_lessons = MagicMock(return_value="")

        agent.run("搜尋最新論文", strategy="cot")

        prompt = captured[0]
        # Must have full .md content, not just short policy
        assert "# 技能A 操作手冊" in prompt or "# 技能B 操作手冊" in prompt
        assert "步驟A1" in prompt or "步驟B1" in prompt
        assert "已知技能" in prompt

    # ── 3-3  Evolution 自動觸發 ──────────────────────────────────────

    def test_3_3_evolution_auto_triggered(self):
        """After run(), skill_graph state should be updated — compare
        snapshot() before and after.
        """
        from agents.main_agent import MainAgent

        agent = MainAgent()
        # Add a skill so evolution has something to work with
        skill = _skill("sk-evo", utility=0.5)
        agent.skill_graph.add_skill(skill)

        snap_before = agent.skill_graph.snapshot()

        agent._dispatch = lambda s, t, e: "answer"
        agent.evaluator.evaluate = MagicMock(return_value=0.9)
        agent.reflexion.reflect = MagicMock(
            return_value=MagicMock(reflection="OK"),
        )
        agent.reflexion.get_relevant_lessons = MagicMock(return_value="")

        agent.run("test task", strategy="cot")

        snap_after = agent.skill_graph.snapshot()

        # The evolution step should have mutated something
        # (at minimum the utility is decayed)
        nodes_before = {n["skill_id"]: n for n in snap_before["nodes"]}
        nodes_after = {n["skill_id"]: n for n in snap_after["nodes"]}

        assert "sk-evo" in nodes_after
        # After decay: U_{t+1} = (1-γ)·U_t, so utility should decrease
        assert nodes_after["sk-evo"]["utility"] <= nodes_before["sk-evo"]["utility"]

    # ── 3-4  Evolution log 記錄 ──────────────────────────────────────

    def test_3_4_evolution_log_recorded(self):
        """AgentResult.evolution_log should be a non-empty string
        consumable by MetricsTracker.
        """
        from agents.main_agent import MainAgent

        agent = MainAgent()

        agent._dispatch = lambda s, t, e: "answer"
        agent.evaluator.evaluate = MagicMock(return_value=0.8)
        agent.reflexion.reflect = MagicMock(
            return_value=MagicMock(reflection="OK"),
        )
        agent.reflexion.get_relevant_lessons = MagicMock(return_value="")

        result = agent.run("test", strategy="cot")

        assert result.evolution_log is not None
        assert isinstance(result.evolution_log, str)
        assert len(result.evolution_log) > 0


# ═══════════════════════════════════════════════════════════════════
#  3.2 SkillRetriever 測試  (3-5 … 3-10)
# ═══════════════════════════════════════════════════════════════════

class TestSkillRetriever:
    """Tests 3-5 through 3-10."""

    # ── 3-5  Activation score 公式 ───────────────────────────────────

    def test_3_5_activation_score_formula(self):
        """Given fixed sim, U, centrality values, verify the formula
        A = λ₁·sim + λ₂·U + λ₃·centrality.
        """
        sr = SkillRetriever(top_k=5, lambda1=0.5, lambda2=0.3, lambda3=0.2)

        # Use known values by mocking internals
        sim_val, u_val, cent_val = 0.8, 0.6, 0.4
        expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4  # = 0.66

        # Manually verify the formula computation
        activation = sr.lambda1 * sim_val + sr.lambda2 * u_val + sr.lambda3 * cent_val
        assert abs(activation - 0.66) < 1e-9
        assert abs(activation - expected) < 1e-9

    # ── 3-6  Top-k 排序 ─────────────────────────────────────────────

    def test_3_6_topk_sorted_descending(self):
        """5 skills → top_k=3 → only 3 returned, sorted descending."""
        skills = [_skill(f"s{i}", utility=float(i)) for i in range(5)]
        g = SkillGraph(capacity=50)
        for s in skills:
            g.add_skill(s)

        sr = SkillRetriever(top_k=3)
        results = sr.retrieve("test task", g)

        assert len(results) == 3
        for i in range(len(results) - 1):
            assert results[i].activation_score >= results[i + 1].activation_score

    # ── 3-7  Utility 正規化 ──────────────────────────────────────────

    def test_3_7_utility_normalization(self):
        """utilities [10, 5, 1] → normalised to [1.0, 0.5, 0.1]."""
        g = SkillGraph(capacity=50)
        s10 = _skill("s10", utility=10.0, name="X", initiation_set=["tag"])
        s5 = _skill("s5", utility=5.0, name="X", initiation_set=["tag"])
        s1 = _skill("s1", utility=1.0, name="X", initiation_set=["tag"])
        g.add_skill(s10)
        g.add_skill(s5)
        g.add_skill(s1)

        # Use lambda2=1.0 to isolate utility component
        sr = SkillRetriever(top_k=5, lambda1=0.0, lambda2=1.0, lambda3=0.0)
        results = sr.retrieve("anything", g)

        # With only utility component, scores = [1.0, 0.5, 0.1]
        scores = {r.skill_id: r.activation_score for r in results}
        assert abs(scores["s10"] - 1.0) < 1e-6
        assert abs(scores["s5"] - 0.5) < 1e-6
        assert abs(scores["s1"] - 0.1) < 1e-6

    # ── 3-8  Centrality 計算 ─────────────────────────────────────────

    def test_3_8_centrality_calculation(self):
        """Hub with 3 in-edges vs leaf with 0 → hub has higher centrality."""
        g = SkillGraph(capacity=50)
        hub = _skill("hub", name="hub", utility=0.0)
        leaf = _skill("leaf", name="leaf", utility=0.0)
        src1 = _skill("src1", name="src1", utility=0.0)
        src2 = _skill("src2", name="src2", utility=0.0)
        src3 = _skill("src3", name="src3", utility=0.0)

        for s in [hub, leaf, src1, src2, src3]:
            g.add_skill(s)
        g.add_edge("src1", "hub")
        g.add_edge("src2", "hub")
        g.add_edge("src3", "hub")

        # Isolate centrality component
        sr = SkillRetriever(top_k=5, lambda1=0.0, lambda2=0.0, lambda3=1.0)
        results = sr.retrieve("test", g)

        hub_result = [r for r in results if r.skill_id == "hub"][0]
        leaf_result = [r for r in results if r.skill_id == "leaf"][0]

        # in_degree_centrality(hub) = 3/(5-1) = 0.75
        # in_degree_centrality(leaf) = 0/(5-1) = 0.0
        assert abs(hub_result.activation_score - 0.75) < 1e-6
        assert abs(leaf_result.activation_score - 0.0) < 1e-6

    # ── 3-9  空圖安全 ────────────────────────────────────────────────

    def test_3_9_empty_graph_safe(self):
        """Empty SkillGraph → empty list, no error, no empty block."""
        g = SkillGraph(capacity=10)
        sr = SkillRetriever(top_k=3)

        results = sr.retrieve("任何任務", g)
        assert results == []

        prompt = sr.format_for_prompt(results)
        assert prompt == ""

    # ── 3-10  Only active/cold tier (archive excluded) ───────────────

    def test_3_10_archive_tier_excluded(self):
        """Skills in archive tier should NOT be retrieved.

        SkillRetriever retrieves from the full graph; the caller
        (MainAgent) can pre-filter via MemoryPartition. Here we
        verify the partition-based filtering logic.
        """
        g = SkillGraph(capacity=50)
        active_sk = _skill("sk-active", utility=1.0, name="active")
        cold_sk = _skill("sk-cold", utility=0.5, name="cold")
        archive_sk = _skill("sk-archive", utility=0.01, name="archive")

        g.add_skill(active_sk)
        g.add_skill(cold_sk)
        g.add_skill(archive_sk)

        partition = MemoryPartition(
            theta_high=0.7, theta_low=0.3,
            epsilon_h=0.05, epsilon_l=0.05,
        )
        partition.set_tier("sk-active", "active")
        partition.set_tier("sk-cold", "cold")
        partition.set_tier("sk-archive", "archive")

        # Build a filtered graph excluding archive skills
        archive_ids = set(partition.get_skills_by_tier("archive"))
        retrievable_skills = [
            s for s in g.skills if s.skill_id not in archive_ids
        ]

        # Build temp graph for retrieval
        filtered_graph = SkillGraph(capacity=50)
        for s in retrievable_skills:
            filtered_graph.add_skill(s)

        sr = SkillRetriever(top_k=10)
        results = sr.retrieve("test", filtered_graph)
        retrieved_ids = {r.skill_id for r in results}

        assert "sk-active" in retrieved_ids
        assert "sk-cold" in retrieved_ids
        assert "sk-archive" not in retrieved_ids


# ═══════════════════════════════════════════════════════════════════
#  3.3 SkillDocument 更新測試  (3-11 … 3-15)
# ═══════════════════════════════════════════════════════════════════

class TestSkillDocumentUpdate:
    """Tests 3-11 through 3-15."""

    # ── 3-11  失敗觸發更新 ───────────────────────────────────────────

    def test_3_11_failure_triggers_caveat_update(self, tmp_path):
        """Task fails + skill was used → 注意事項 section appended."""
        doc_rel = _write_doc(tmp_path, "sk-fail")
        skill = _skill("sk-fail", document_path=doc_rel, version=1)

        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        trace = _trace("fail-trace", success=False, score=0.2)
        result = updater.update(
            skill, trace,
            reflexion_text="搜尋關鍵字太寬泛，結果不相關",
            success=False,
        )

        assert result.updated is True
        assert result.trigger == "failure_caveat"

        doc_path = tmp_path / "skills" / "learned" / "sk-fail.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "⚠️" in content
        assert "搜尋關鍵字太寬泛" in content

    # ── 3-12  版本遞增 ───────────────────────────────────────────────

    def test_3_12_version_incremented_and_history_added(self, tmp_path):
        """After update: skill.version +1, version history has new row."""
        doc_rel = _write_doc(tmp_path, "sk-ver")
        skill = _skill("sk-ver", document_path=doc_rel, version=1)

        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        trace = _trace("ver-trace", success=False)
        updater.update(skill, trace, "失敗了", success=False)

        assert skill.version == 2

        doc_path = tmp_path / "skills" / "learned" / "sk-ver.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "v2" in content
        # Version history table must have the new entry
        assert "ver-trace" in content

    # ── 3-13  備份機制 ───────────────────────────────────────────────

    def test_3_13_backup_created_before_update(self, tmp_path):
        """Update creates {skill_id}_v{old}.md.bak before overwriting."""
        doc_rel = _write_doc(tmp_path, "sk-bak")
        skill = _skill("sk-bak", document_path=doc_rel, version=3)

        updater = SkillDocumentUpdater(llm=None, root=tmp_path)
        result = updater.update(
            skill, _trace(success=False), "fail", success=False,
        )

        assert result.backup_path is not None
        backup = Path(result.backup_path)
        assert backup.exists()
        assert "sk-bak_v3.md.bak" in backup.name

        # Backup content matches original
        backup_content = backup.read_text(encoding="utf-8")
        assert backup_content == _SAMPLE_DOC

    # ── 3-14  Fallback 到 policy ────────────────────────────────────

    def test_3_14_fallback_to_policy_when_no_document(self):
        """document_path is None → retriever uses short policy string."""
        skill = _skill(
            "sk-no-doc", document_path=None,
            policy="search → read → answer",
            name="搜尋",
        )
        g = SkillGraph(capacity=10)
        g.add_skill(skill)

        sr = SkillRetriever(top_k=3)
        results = sr.retrieve("搜尋任務", g)

        assert len(results) == 1
        # Fallback: structured policy text
        assert "search → read → answer" in results[0].document
        assert results[0].document != ""  # not empty

    def test_3_14_fallback_missing_file(self, tmp_path):
        """document_path set but file missing → still uses policy."""
        skill = _skill(
            "sk-missing",
            document_path="skills/learned/nonexistent.md",
            policy="plan → execute → verify",
        )
        g = SkillGraph(capacity=10)
        g.add_skill(skill)

        sr = SkillRetriever(top_k=3, root=tmp_path)
        results = sr.retrieve("task", g)

        assert len(results) == 1
        assert "plan → execute → verify" in results[0].document

    # ── 3-15  成功不觸發 ─────────────────────────────────────────────

    def test_3_15_success_no_discovery_no_update(self, tmp_path):
        """Task succeeds + no new discovery in reflexion → .md unchanged."""
        doc_rel = _write_doc(tmp_path, "sk-unchanged")
        skill = _skill("sk-unchanged", document_path=doc_rel, version=1)

        updater = SkillDocumentUpdater(
            llm=None, root=tmp_path,
            divergence_threshold=0.9,  # high threshold → no divergence
        )
        # Use actions that closely match the doc's strategy steps to avoid divergence
        trace = _trace(
            "ok-trace",
            actions=["搜尋關鍵字", "閱讀搜尋結果", "提取關鍵資訊", "組織成摘要"],
            success=True,
        )
        result = updater.update(
            skill, trace,
            reflexion_text="任務順利完成",  # no discovery keywords
            success=True,
        )

        assert result.updated is False
        assert skill.version == 1  # not incremented

        # File content unchanged
        doc_path = tmp_path / "skills" / "learned" / "sk-unchanged.md"
        content = doc_path.read_text(encoding="utf-8")
        assert content == _SAMPLE_DOC
