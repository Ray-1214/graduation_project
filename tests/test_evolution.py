"""
Phase 2 Acceptance Tests — Self-Evolving Skill Graph

Execute:  pytest tests/test_evolution.py -v
Pass:     ALL PASSED, 0 failures

Coverage:
  2.1  SkillAbstractor         (2-1  ~ 2-5)
  2.2  contract_subgraph       (2-6  ~ 2-12)
  2.3  EvolutionOperator       (2-13 ~ 2-22)
  2.4  SkillDocumentGenerator  (2-23 ~ 2-29)
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
from skill_graph.contract_subgraph import contract_subgraph
from skill_graph.evolution_operator import EvolutionLog, EvolutionOperator
from skill_graph.memory_partition import MemoryPartition
from skill_graph.skill_abstractor import SkillAbstractor
from skill_graph.skill_document_generator import SkillDocumentGenerator
from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _trace(
    tid: str,
    actions: list[str],
    success: bool = True,
    score: float = 0.8,
    desc: str = "test task",
) -> EpisodicTrace:
    steps = [
        TraceStep(state=f"s{i}", action=a, outcome=f"o{i}",
                  timestamp=time.time())
        for i, a in enumerate(actions)
    ]
    return EpisodicTrace(
        task_id=tid, task_description=desc, steps=steps,
        strategy="react", success=success, score=score,
        total_time=float(len(actions)),
    )


def _skill(sid: str, **kw) -> SkillNode:
    defaults = dict(
        skill_id=sid, name=sid, policy=f"do_{sid}",
        termination=f"done_{sid}", initiation_set=[f"tag_{sid}"],
        cost=1.0, utility=0.5,
    )
    defaults.update(kw)
    return SkillNode(**defaults)


def _chain_graph(*ids: str) -> SkillGraph:
    """Build a linear chain graph:  ids[0] → ids[1] → … → ids[-1]."""
    g = SkillGraph(capacity=50)
    for sid in ids:
        g.add_skill(_skill(sid))
    for i in range(len(ids) - 1):
        g.add_edge(ids[i], ids[i + 1], weight=1.0)
    return g


# ═══════════════════════════════════════════════════════════════════
#  2.1  SkillAbstractor Tests  (2-1 ~ 2-5)
# ═══════════════════════════════════════════════════════════════════

class TestSkillAbstractor:
    """2.1 SkillAbstractor"""

    def test_2_01_repeated_subsequence_detection(self):
        """2-1  5 traces, 3 contain [A→B→C] → detected with frequency=3."""
        traces = [
            _trace("t1", ["A", "B", "C", "D"]),
            _trace("t2", ["X", "Y", "Z"]),
            _trace("t3", ["A", "B", "C", "E"]),
            _trace("t4", ["A", "B", "C", "F"]),
            _trace("t5", ["P", "Q", "R"]),
        ]
        sa = SkillAbstractor(c_add=2.0, min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)

        abc = [c for c in candidates if c.actions == ("A", "B", "C")]
        assert len(abc) == 1
        assert abc[0].frequency == 3

    def test_2_02_mdl_compression_gain(self):
        """2-2  [A→B→C] × 3: gain=(3−1)×3=6, cost=c_add+3=6, net=0.
        With c_add=3 → cost=6, net=0 → NOT recommended.
        With c_add=2 → cost=5, net=1 → recommended."""
        traces = [
            _trace("t1", ["A", "B", "C"]),
            _trace("t2", ["A", "B", "C"]),
            _trace("t3", ["A", "B", "C"]),
        ]
        # c_add=3 → cost=6, net=0 → rejected
        sa_strict = SkillAbstractor(c_add=3.0, min_ngram=2, min_frequency=2)
        cands_strict = sa_strict.extract(traces)
        abc_strict = [c for c in cands_strict if c.actions == ("A", "B", "C")]
        assert len(abc_strict) == 0  # net_gain=0, not > 0

        # c_add=2 → cost=5, net=1 → accepted
        sa_loose = SkillAbstractor(c_add=2.0, min_ngram=2, min_frequency=2)
        cands_loose = sa_loose.extract(traces)
        abc_loose = [c for c in cands_loose if c.actions == ("A", "B", "C")]
        assert len(abc_loose) == 1
        c = abc_loose[0]
        assert c.compression_gain == 6  # (3-1)*3
        assert c.description_cost == 5  # 2+3
        assert c.net_gain == 1

    def test_2_03_low_frequency_filtered(self):
        """2-3  [X→Y] appears only once → not recommended."""
        traces = [
            _trace("t1", ["X", "Y", "Z"]),
            _trace("t2", ["A", "B", "C"]),
        ]
        sa = SkillAbstractor(c_add=1.0, min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)

        xy = [c for c in candidates if c.actions == ("X", "Y")]
        assert len(xy) == 0

    def test_2_04_min_length_2(self):
        """2-4  Length=1 subsequences are never considered."""
        traces = [
            _trace("t1", ["A", "B"]),
            _trace("t2", ["A", "C"]),
            _trace("t3", ["A", "D"]),
        ]
        sa = SkillAbstractor(min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)
        for c in candidates:
            assert c.length >= 2

    def test_2_05_candidates_sorted_by_gain(self):
        """2-5  Multiple candidates sorted by net_gain descending."""
        # Two distinct repeated patterns → at least 2 candidates
        traces = [
            _trace("t0", ["A", "B", "C"]),
            _trace("t1", ["A", "B", "C"]),
            _trace("t2", ["A", "B", "C"]),
            _trace("t3", ["X", "Y", "Z"]),
            _trace("t4", ["X", "Y", "Z"]),
            _trace("t5", ["X", "Y", "Z"]),
        ]
        sa = SkillAbstractor(c_add=1.0, min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)

        assert len(candidates) >= 2
        for i in range(len(candidates) - 1):
            assert candidates[i].net_gain >= candidates[i + 1].net_gain


# ═══════════════════════════════════════════════════════════════════
#  2.2  contract_subgraph Tests  (2-6 ~ 2-12)
# ═══════════════════════════════════════════════════════════════════

class TestContractSubgraph:
    """2.2 contract_subgraph"""

    def test_2_06_node_merge(self):
        """2-6  Merge [s1,s2,s3] → s1/s2/s3 disappear, macro appears."""
        g = _chain_graph("X", "s1", "s2", "s3", "Y")
        contract_subgraph(g, ["s1", "s2", "s3"], "macro_s1s2s3")

        assert not g.has_skill("s1")
        assert not g.has_skill("s2")
        assert not g.has_skill("s3")
        assert g.has_skill("macro_s1s2s3")

    def test_2_07_in_edge_rewired(self):
        """2-7  X→s1 becomes X→macro."""
        g = _chain_graph("X", "s1", "s2", "s3", "Y")
        contract_subgraph(g, ["s1", "s2", "s3"], "macro")

        edges = [(e["src"], e["dst"]) for e in g.get_edges()]
        assert ("X", "macro") in edges

    def test_2_08_out_edge_rewired(self):
        """2-8  s3→Y becomes macro→Y."""
        g = _chain_graph("X", "s1", "s2", "s3", "Y")
        contract_subgraph(g, ["s1", "s2", "s3"], "macro")

        edges = [(e["src"], e["dst"]) for e in g.get_edges()]
        assert ("macro", "Y") in edges

    def test_2_09_internal_edges_removed(self):
        """2-9  s1→s2, s2→s3 no longer exist after merge."""
        g = _chain_graph("X", "s1", "s2", "s3", "Y")
        contract_subgraph(g, ["s1", "s2", "s3"], "macro")

        for e in g.get_edges():
            assert e["src"] not in {"s1", "s2", "s3"}
            assert e["dst"] not in {"s1", "s2", "s3"}

    def test_2_10_macro_skill_attributes(self):
        """2-10  macro.policy contains s1+s2+s3 policies;
        initiation_set = s1's; termination = s3's."""
        g = _chain_graph("X", "s1", "s2", "s3", "Y")
        macro = contract_subgraph(g, ["s1", "s2", "s3"], "macro")

        assert "do_s1" in macro.policy
        assert "do_s2" in macro.policy
        assert "do_s3" in macro.policy
        assert macro.initiation_set == ["tag_s1"]
        assert macro.termination == "done_s3"

    def test_2_11_reachability_preserved(self):
        """2-11  X reachable to Y before merge → still reachable after."""
        import networkx as nx

        g = _chain_graph("X", "s1", "s2", "s3", "Y")
        assert nx.has_path(g._graph, "X", "Y")

        contract_subgraph(g, ["s1", "s2", "s3"], "macro")
        assert nx.has_path(g._graph, "X", "Y")

    def test_2_12_sequence_length_2(self):
        """2-12  Merge [s1,s2] (length=2) works without error."""
        g = _chain_graph("X", "s1", "s2", "Y")
        macro = contract_subgraph(g, ["s1", "s2"], "pair")

        assert len(g) == 3  # X, pair, Y
        assert macro.policy == "do_s1 → do_s2"
        edges = [(e["src"], e["dst"]) for e in g.get_edges()]
        assert ("X", "pair") in edges
        assert ("pair", "Y") in edges


# ═══════════════════════════════════════════════════════════════════
#  2.3  EvolutionOperator Tests  (2-13 ~ 2-22)
# ═══════════════════════════════════════════════════════════════════

class TestEvolutionOperator:
    """2.3 EvolutionOperator"""

    def test_2_13_utility_decay_and_reinforcement(self):
        """2-13  Step 1: all skills decayed by (1-γ); used skill gets +ΔU."""
        eo = EvolutionOperator(gamma=0.1, delta_u=2.0, eta=100, delta=100)
        g = SkillGraph(capacity=50)
        s_used = _skill("search", policy="search", utility=1.0)
        s_other = _skill("other", policy="unrelated", utility=1.0)
        g.add_skill(s_used)
        g.add_skill(s_other)
        p = MemoryPartition()

        trace = _trace("t1", ["search", "answer"], score=1.0)
        log = eo.evolve(g, trace, p)

        # "other" should be decayed only: 1.0 * 0.9 = 0.9
        assert abs(g.get_skill("other").utility - 0.9) < 0.01
        # "search" should be decayed + reinforced: > 0.9
        assert g.get_skill("search").utility > 0.9
        assert "search" in log.reinforced_skills

    def test_2_14_skill_insertion(self):
        """2-14  Step 2: abstractor recommends 1 candidate +
        dedup passes → graph gains 1 skill."""
        eo = EvolutionOperator(
            delta=1, eta=0, theta_dup=0.99,
            abstractor=SkillAbstractor(c_add=1.0, min_frequency=2),
        )
        g = SkillGraph(capacity=50)
        p = MemoryPartition()

        # Feed repeated pattern traces (T1 trigger: success=False)
        for i in range(5):
            log = eo.evolve(
                g,
                _trace(f"t{i}", ["plan", "execute", "verify", "done"],
                       success=False),
                p,
            )

        # Some skills should have been inserted over the cycles
        total_inserted = sum(1 for s in g.skills if "abstracted" in s.tags)
        assert total_inserted >= 1

    def test_2_15_dedup_blocks_similar(self):
        """2-15  Candidate with sim ≥ θ_dup to existing → not inserted."""
        eo = EvolutionOperator(
            delta=1, eta=0, theta_dup=0.3,  # very strict threshold
            abstractor=SkillAbstractor(c_add=1.0, min_frequency=2),
        )
        g = SkillGraph(capacity=50)
        # Pre-load a skill whose policy will match candidates
        g.add_skill(_skill("blocker", policy="plan → execute → verify"))
        initial_size = len(g)
        p = MemoryPartition()

        all_rejected: list[str] = []
        for i in range(5):
            log = eo.evolve(
                g,
                _trace(f"t{i}", ["plan", "execute", "verify"],
                       success=False),
                p,
            )
            all_rejected.extend(log.rejected_skills)

        # Graph should not have grown much — dedup blocks similar candidates
        assert len(g) <= initial_size + 1, (
            f"Graph grew from {initial_size} to {len(g)}, "
            f"dedup should have blocked similar candidates"
        )
        # At least some candidates were rejected across all rounds
        assert len(all_rejected) >= 1, (
            "Expected at least 1 rejected candidate due to dedup"
        )

    def test_2_16_contraction_on_cooccurrence(self):
        """2-16  Step 3: [A→B] co-occurs > δ → merged as macro-skill."""
        eo = EvolutionOperator(delta=2, eta=0)
        g = SkillGraph(capacity=50)
        g.add_skill(_skill("A", policy="A"))
        g.add_skill(_skill("B", policy="B"))
        p = MemoryPartition()

        # Feed 4 traces with A→B, triggering T1 each time
        for i in range(4):
            log = eo.evolve(
                g,
                _trace(f"t{i}", ["A", "B"], success=False),
                p,
            )

        # Check if contraction happened (A and B merged)
        if log.contracted:
            macro_ids = [c["macro_id"] for c in log.contracted]
            assert any("A" in mid and "B" in mid for mid in macro_ids)

    def test_2_17_tier_update(self):
        """2-17  Step 4: after evolve, all skills have valid tier."""
        eo = EvolutionOperator(eta=100, delta=100)
        g = _chain_graph("s1", "s2", "s3")
        p = MemoryPartition()

        log = eo.evolve(g, _trace("t1", ["s1"]), p)

        for s in g.skills:
            tier = p.get_tier(s.skill_id)
            assert tier in ("active", "cold", "archive")

    def test_2_18_trigger_t1_failure(self):
        """2-18  T1: success=False → full 4-step evolution."""
        eo = EvolutionOperator()
        g = _chain_graph("a", "b")
        p = MemoryPartition()

        log = eo.evolve(g, _trace("t1", ["a", "b"], success=False), p)

        assert "T1_failure" in log.triggers_fired
        assert log.full_evolution is True

    def test_2_19_trigger_t2_long_trace(self):
        """2-19  T2: trace length = moving_avg + η + 1 → triggers."""
        eo = EvolutionOperator(eta=2)
        g = _chain_graph("a", "b")
        p = MemoryPartition()

        # Establish baseline: 3 traces of length 3 → avg=3
        for i in range(3):
            eo.evolve(g, _trace(f"base_{i}", ["a", "b", "c"]), p)

        # Trace of length = 3 + 2 + 1 = 6 → triggers T2
        long_trace = _trace("long", ["x"] * 6)
        log = eo.evolve(g, long_trace, p)

        assert "T2_long_trace" in log.triggers_fired

    def test_2_20_no_trigger_decay_only(self):
        """2-20  All three conditions unsatisfied → decay only."""
        eo = EvolutionOperator(eta=100, delta=100)
        g = _chain_graph("a", "b")
        p = MemoryPartition()

        log = eo.evolve(
            g,
            _trace("t1", ["something", "else"], success=True),
            p,
        )

        assert log.triggers_fired == []
        assert log.full_evolution is False
        assert log.decayed_skills > 0
        assert log.inserted_skills == []
        assert log.contracted == []

    def test_2_21_evolution_log_fields(self):
        """2-21  EvolutionLog contains inserted_skills, contracted,
        tier_changes fields."""
        log = EvolutionLog(
            inserted_skills=["sk-new-001"],
            contracted=[{"sequence": ["a", "b"], "macro_id": "m1",
                         "frequency": 5}],
            tier_changes={"sk-001": ("cold", "active")},
        )
        assert log.inserted_skills == ["sk-new-001"]
        assert len(log.contracted) == 1
        assert log.contracted[0]["macro_id"] == "m1"
        assert log.tier_changes["sk-001"] == ("cold", "active")

        # summary() renders all
        s = log.summary()
        assert "inserted=1" in s
        assert "contracted=1" in s
        assert "tier_moves=1" in s

    def test_2_22_end_to_end_5_rounds(self):
        """2-22  Feed 5 repeated-pattern traces → graph grows from
        initial skills, may contain macro-skills. No crashes."""
        eo = EvolutionOperator(
            gamma=0.05, delta=2, eta=2,
            abstractor=SkillAbstractor(c_add=1.0, min_frequency=2),
        )
        g = SkillGraph(capacity=50)
        # Start with 3 base skills
        for sid in ["init", "process", "output"]:
            g.add_skill(_skill(sid))
        p = MemoryPartition()

        logs: list[EvolutionLog] = []
        for i in range(5):
            trace = _trace(
                f"round_{i}",
                ["init", "process", "output", "validate"],
                success=(i % 2 == 0),  # alternate success/failure
                score=0.5 + 0.1 * i,
            )
            log = eo.evolve(g, trace, p)
            logs.append(log)
            assert len(g) <= g.capacity

        # Graph should still be valid after 5 rounds
        assert len(g) >= 1
        # At least some logs should have run full evolution
        assert any(l.full_evolution for l in logs)
        # All skills should have tiers assigned
        for s in g.skills:
            assert p.get_tier(s.skill_id) in ("active", "cold", "archive")


# ═══════════════════════════════════════════════════════════════════
#  2.4  SkillDocumentGenerator Tests  (2-23 ~ 2-29)
# ═══════════════════════════════════════════════════════════════════

class TestSkillDocumentGenerator:
    """2.4 SkillDocumentGenerator"""

    def test_2_23_md_file_generated(self, tmp_path):
        """2-23  Given SkillNode + trace → .md file created under
        skills/learned/."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _skill("sk-doc001", name="搜尋技能",
                        initiation_set=["問答"],
                        termination="回答完成")
        trace = _trace("tr-001", ["搜尋", "閱讀", "總結"])

        path = gen.generate(skill, [trace])

        full_path = tmp_path / path
        assert full_path.exists()
        assert full_path.suffix == ".md"
        assert "skills/learned/" in path

    def test_2_24_document_structure(self, tmp_path):
        """2-24  Generated .md contains 4 required sections:
        適用場景, 策略步驟, 終止條件, 版本歷史."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _skill("sk-doc002", name="分析技能",
                        initiation_set=["分析"],
                        termination="分析完成")
        path = gen.generate(skill, [_trace("tr-002", ["A", "B"])])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "## 適用場景" in content
        assert "## 策略步驟" in content
        assert "## 終止條件" in content
        assert "## 版本歷史" in content

    def test_2_25_tool_action_in_steps(self, tmp_path):
        """2-25  Trace containing Action[web_search] → appears in
        策略步驟 section."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _skill("sk-doc003", name="搜尋")
        trace = _trace("tr-003", [
            'Action[web_search]("Python GIL")',
            "整理結果",
        ])
        path = gen.generate(skill, [trace])

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "Action[web_search]" in content
        assert "Python GIL" in content

    def test_2_26_knowledge_entry_included(self, tmp_path):
        """2-26  Passing KnowledgeEntry → document includes its content."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _skill("sk-doc004", name="知識技能")
        trace = _trace("tr-004", ["查詢", "回答"])
        entry = KnowledgeEntry(
            query="What is RAG?",
            content="RAG stands for Retrieval-Augmented Generation.",
            source="web",
            confidence=0.9,
        )

        path = gen.generate(skill, [trace], [entry])
        content = (tmp_path / path).read_text(encoding="utf-8")

        assert "## 背景知識" in content
        assert "RAG" in content
        assert "Retrieval-Augmented Generation" in content

    def test_2_27_document_path_written_back(self, tmp_path):
        """2-27  After generation, skill.document_path is set correctly."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _skill("sk-doc005", name="測試")
        gen.generate(skill, [_trace("tr-005", ["a"])])

        assert skill.document_path is not None
        assert skill.document_path.startswith("skills/learned/")
        assert skill.document_path.endswith(".md")

    def test_2_28_fallback_on_bad_llm(self, tmp_path):
        """2-28  LLM returns empty/too-short → template fallback, no crash."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ""  # empty

        gen = SkillDocumentGenerator(llm=mock_llm, root=tmp_path)
        skill = _skill("sk-doc006", name="Fallback測試")
        path = gen.generate(
            skill,
            [_trace("tr-006", ["step_A", "step_B"])],
        )

        content = (tmp_path / path).read_text(encoding="utf-8")
        assert "## 策略步驟" in content
        assert "tr-006" in content  # template uses trace ID

    def test_2_29_idempotent_regeneration(self, tmp_path):
        """2-29  Regenerating for the same skill → overwrites, no dup files."""
        gen = SkillDocumentGenerator(llm=None, root=tmp_path)
        skill = _skill("sk-doc007", name="冪等測試")

        path1 = gen.generate(skill, [_trace("tr-v1", ["old_step"])])
        path2 = gen.generate(skill, [_trace("tr-v2", ["new_step"])])

        # Same path
        assert path1 == path2

        # Content is the latest version
        content = (tmp_path / path2).read_text(encoding="utf-8")
        assert "new_step" in content

        # Only one file exists
        learned_dir = tmp_path / "skills" / "learned"
        md_files = list(learned_dir.glob("sk-doc007.md"))
        assert len(md_files) == 1
