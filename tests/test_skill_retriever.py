"""
Tests for SkillRetriever.

Covers:
  - Activation score computation (sim, utility, centrality)
  - Top-k limiting
  - .md document loading + fallback
  - Prompt formatting
  - Edge cases (empty graph, λ validation)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.episodic_log import EpisodicTrace, TraceStep
from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode
from skill_graph.skill_retriever import RetrievedSkill, SkillRetriever


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _skill(sid: str, **kw) -> SkillNode:
    defaults = dict(
        skill_id=sid, name=sid, policy=f"do_{sid}",
        termination=f"done_{sid}", initiation_set=[f"tag_{sid}"],
        cost=1.0, utility=0.5,
    )
    defaults.update(kw)
    return SkillNode(**defaults)


def _graph_with_skills(*skills: SkillNode) -> SkillGraph:
    g = SkillGraph(capacity=50)
    for s in skills:
        g.add_skill(s)
    return g


# ═══════════════════════════════════════════════════════════════════
#  Activation Score / Top-k
# ═══════════════════════════════════════════════════════════════════

class TestActivationScore:

    def test_empty_graph_returns_empty(self):
        """Empty graph → empty list."""
        sr = SkillRetriever(top_k=3)
        g = SkillGraph(capacity=10)
        assert sr.retrieve("solve math problem", g) == []

    def test_single_skill_returned(self):
        """Graph with 1 skill → 1 result."""
        sr = SkillRetriever(top_k=3)
        g = _graph_with_skills(_skill("math", name="數學解題"))
        results = sr.retrieve("solve math problem", g)
        assert len(results) == 1
        assert results[0].skill_id == "math"
        assert results[0].activation_score > 0

    def test_top_k_limits_results(self):
        """More skills than top_k → only top_k returned."""
        skills = [_skill(f"s{i}", utility=float(i)) for i in range(10)]
        g = _graph_with_skills(*skills)
        sr = SkillRetriever(top_k=3)
        results = sr.retrieve("test task", g)
        assert len(results) == 3

    def test_sorted_descending(self):
        """Results are sorted by activation score descending."""
        skills = [
            _skill("high", name="high relevance", utility=1.0,
                   initiation_set=["test"]),
            _skill("low", name="low relevance", utility=0.1,
                   initiation_set=["unrelated"]),
        ]
        g = _graph_with_skills(*skills)
        sr = SkillRetriever(top_k=5)
        results = sr.retrieve("test task", g)
        for i in range(len(results) - 1):
            assert results[i].activation_score >= results[i + 1].activation_score

    def test_utility_influences_score(self):
        """Higher utility → higher activation (all else equal)."""
        s_high = _skill("high_u", name="X", policy="do_x",
                        utility=1.0, initiation_set=["tag"])
        s_low = _skill("low_u", name="X", policy="do_x",
                       utility=0.01, initiation_set=["tag"])
        g = _graph_with_skills(s_high, s_low)
        sr = SkillRetriever(top_k=5, lambda1=0.0, lambda2=1.0, lambda3=0.0)
        results = sr.retrieve("anything", g)
        assert results[0].skill_id == "high_u"

    def test_centrality_influences_score(self):
        """Skill with more in-edges → higher centrality component."""
        g = SkillGraph(capacity=50)
        g.add_skill(_skill("hub", name="hub node"))
        g.add_skill(_skill("leaf", name="leaf node"))
        g.add_skill(_skill("src1", name="source 1"))
        g.add_skill(_skill("src2", name="source 2"))
        # hub gets 2 in-edges, leaf gets 0
        g.add_edge("src1", "hub")
        g.add_edge("src2", "hub")

        sr = SkillRetriever(top_k=5, lambda1=0.0, lambda2=0.0, lambda3=1.0)
        results = sr.retrieve("test", g)
        hub_r = [r for r in results if r.skill_id == "hub"][0]
        leaf_r = [r for r in results if r.skill_id == "leaf"][0]
        assert hub_r.activation_score > leaf_r.activation_score

    def test_lambda_validation(self):
        """λ weights not summing to 1.0 → ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            SkillRetriever(lambda1=0.5, lambda2=0.5, lambda3=0.5)


# ═══════════════════════════════════════════════════════════════════
#  Document Loading
# ═══════════════════════════════════════════════════════════════════

class TestDocumentLoading:

    def test_md_document_loaded(self, tmp_path):
        """When document_path points to valid .md → full content used."""
        # Create the document
        doc_dir = tmp_path / "skills" / "learned"
        doc_dir.mkdir(parents=True)
        doc_file = doc_dir / "sk-doc.md"
        doc_file.write_text("# 搜尋技能\n\n## 策略步驟\n1. 搜尋\n2. 總結",
                           encoding="utf-8")

        skill = _skill("sk-doc", document_path="skills/learned/sk-doc.md")
        g = _graph_with_skills(skill)

        sr = SkillRetriever(top_k=3, root=tmp_path)
        results = sr.retrieve("search task", g)

        assert len(results) == 1
        assert "# 搜尋技能" in results[0].document
        assert "策略步驟" in results[0].document

    def test_missing_md_falls_back_to_policy(self, tmp_path):
        """When document_path file doesn't exist → fallback to policy."""
        skill = _skill("sk-missing",
                       document_path="skills/learned/nonexistent.md",
                       policy="plan → execute")
        g = _graph_with_skills(skill)

        sr = SkillRetriever(top_k=3, root=tmp_path)
        results = sr.retrieve("task", g)

        assert "plan → execute" in results[0].document
        assert "# " not in results[0].document  # no markdown header

    def test_no_document_path_uses_policy(self):
        """When document_path is None → fallback to structured policy."""
        skill = _skill("sk-none", policy="search → read → summarize",
                       name="搜尋總結")
        g = _graph_with_skills(skill)

        sr = SkillRetriever(top_k=3)
        results = sr.retrieve("task", g)

        assert "search → read → summarize" in results[0].document
        assert "搜尋總結" in results[0].document

    def test_empty_md_falls_back(self, tmp_path):
        """Empty .md file → fallback to policy."""
        doc_dir = tmp_path / "skills" / "learned"
        doc_dir.mkdir(parents=True)
        doc_file = doc_dir / "empty.md"
        doc_file.write_text("", encoding="utf-8")

        skill = _skill("sk-empty", document_path="skills/learned/empty.md",
                       policy="fallback_policy")
        g = _graph_with_skills(skill)

        sr = SkillRetriever(top_k=3, root=tmp_path)
        results = sr.retrieve("task", g)
        assert "fallback_policy" in results[0].document


# ═══════════════════════════════════════════════════════════════════
#  Prompt Formatting
# ═══════════════════════════════════════════════════════════════════

class TestPromptFormatting:

    def test_format_includes_skill_name(self):
        """Formatted output contains skill name and activation score."""
        sr = SkillRetriever()
        retrieved = [
            RetrievedSkill(
                skill_id="sk-1", name="搜尋技能", policy="search",
                activation_score=0.87, document="test document content",
            ),
        ]
        prompt = sr.format_for_prompt(retrieved)
        assert "已知技能：搜尋技能" in prompt
        assert "0.87" in prompt
        assert "test document content" in prompt

    def test_format_empty_returns_empty_string(self):
        """No retrieved skills → empty string."""
        sr = SkillRetriever()
        assert sr.format_for_prompt([]) == ""

    def test_format_multiple_skills(self):
        """Multiple skills → multiple blocks."""
        sr = SkillRetriever()
        retrieved = [
            RetrievedSkill("s1", "技能A", "p1", 0.9, "doc A"),
            RetrievedSkill("s2", "技能B", "p2", 0.7, "doc B"),
        ]
        prompt = sr.format_for_prompt(retrieved)
        assert "技能A" in prompt
        assert "技能B" in prompt
        assert "doc A" in prompt
        assert "doc B" in prompt

    def test_format_with_full_md_document(self, tmp_path):
        """End-to-end: retrieve → format with .md content."""
        doc_dir = tmp_path / "skills" / "learned"
        doc_dir.mkdir(parents=True)
        doc_file = doc_dir / "sk-full.md"
        md_content = (
            "# 搜尋並總結\n\n"
            "## 適用場景\n- 問答\n\n"
            "## 策略步驟\n1. 搜尋\n2. 閱讀\n3. 總結\n"
        )
        doc_file.write_text(md_content, encoding="utf-8")

        skill = _skill("sk-full", name="搜尋並總結",
                       document_path="skills/learned/sk-full.md")
        g = _graph_with_skills(skill)

        sr = SkillRetriever(top_k=3, root=tmp_path)
        results = sr.retrieve("search task", g)
        prompt = sr.format_for_prompt(results)

        assert "已知技能：搜尋並總結" in prompt
        assert "## 適用場景" in prompt
        assert "## 策略步驟" in prompt


# ═══════════════════════════════════════════════════════════════════
#  repr
# ═══════════════════════════════════════════════════════════════════

class TestRepr:

    def test_repr(self):
        sr = SkillRetriever(top_k=5, lambda1=0.4, lambda2=0.4, lambda3=0.2)
        r = repr(sr)
        assert "top_k=5" in r
        assert "0.4" in r
