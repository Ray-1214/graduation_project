"""
Tests for SkillGraph and MemoryPartition persistence (save/load).

Uses pytest's ``tmp_path`` fixture so no real files are touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode
from skill_graph.memory_partition import MemoryPartition


# ═══════════════════════════════════════════════════════════════════
#  SkillGraph persistence
# ═══════════════════════════════════════════════════════════════════


class TestSkillGraphPersistence:
    """Tests for SkillGraph save/load functionality."""

    # ── save() ────────────────────────────────────────────────────

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """save() should create a valid JSON file."""
        graph = SkillGraph()
        graph.add_skill(SkillNode(
            skill_id="sk-test-001",
            name="Test Skill",
            policy="Do something",
            initiation_set=["math", "計算"],
        ))

        save_path = tmp_path / "graph.json"
        graph.save(save_path)

        assert save_path.exists()
        data = json.loads(save_path.read_text(encoding="utf-8"))
        assert data["num_skills"] == 1
        assert data["num_edges"] == 0
        assert "nodes" in data
        assert "edges" in data

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save() should auto-create missing parent directories."""
        graph = SkillGraph()
        save_path = tmp_path / "nested" / "dir" / "graph.json"
        graph.save(save_path)

        assert save_path.exists()

    # ── load() ────────────────────────────────────────────────────

    def test_load_restores_skills(self, tmp_path: Path) -> None:
        """load() should restore all skill nodes with correct fields."""
        original = SkillGraph()
        original.add_skill(SkillNode(
            skill_id="sk-001",
            name="Skill A",
            policy="Policy A",
            reinforcement=5.0,
            frequency=3,
        ))
        original.add_skill(SkillNode(
            skill_id="sk-002",
            name="Skill B",
            policy="Policy B",
        ))
        original.add_edge("sk-001", "sk-002", weight=0.8, edge_type="dependency")

        save_path = tmp_path / "graph.json"
        original.save(save_path)

        restored = SkillGraph.load(save_path)

        assert len(restored) == 2
        assert restored.has_skill("sk-001")
        assert restored.has_skill("sk-002")

        s1 = restored.get_skill("sk-001")
        assert s1.name == "Skill A"
        assert s1.policy == "Policy A"
        assert s1.reinforcement == 5.0
        assert s1.frequency == 3

    def test_load_restores_edges(self, tmp_path: Path) -> None:
        """load() should restore all edges with correct attributes."""
        original = SkillGraph()
        original.add_skill(SkillNode(skill_id="s1", name="S1", policy="P1"))
        original.add_skill(SkillNode(skill_id="s2", name="S2", policy="P2"))
        original.add_skill(SkillNode(skill_id="s3", name="S3", policy="P3"))
        original.add_edge("s1", "s2", weight=0.5, edge_type="co_occurrence")
        original.add_edge("s2", "s3", weight=0.9, edge_type="abstraction")

        save_path = tmp_path / "graph.json"
        original.save(save_path)
        restored = SkillGraph.load(save_path)

        assert restored._graph.number_of_edges() == 2
        assert restored._graph.has_edge("s1", "s2")
        assert restored._graph.edges["s1", "s2"]["weight"] == 0.5
        assert restored._graph.edges["s1", "s2"]["edge_type"] == "co_occurrence"
        assert restored._graph.has_edge("s2", "s3")
        assert restored._graph.edges["s2", "s3"]["weight"] == 0.9
        assert restored._graph.edges["s2", "s3"]["edge_type"] == "abstraction"

    def test_load_preserves_capacity(self, tmp_path: Path) -> None:
        """load() should accept a custom capacity parameter."""
        graph = SkillGraph(capacity=25)
        graph.add_skill(SkillNode(skill_id="s1", name="S1"))
        save_path = tmp_path / "graph.json"
        graph.save(save_path)

        restored = SkillGraph.load(save_path, capacity=50)
        assert restored.capacity == 50
        assert len(restored) == 1

    # ── Error handling ────────────────────────────────────────────

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """load() should return empty graph if file doesn't exist."""
        missing_path = tmp_path / "nonexistent.json"
        graph = SkillGraph.load(missing_path)

        assert len(graph) == 0
        assert graph._graph.number_of_edges() == 0

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """load() should handle corrupt JSON gracefully."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not valid json {{{", encoding="utf-8")

        graph = SkillGraph.load(bad_path)
        assert len(graph) == 0

    def test_load_empty_json(self, tmp_path: Path) -> None:
        """load() should handle a valid JSON with no nodes/edges."""
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("{}", encoding="utf-8")

        graph = SkillGraph.load(empty_path)
        assert len(graph) == 0

    # ── Round-trip fidelity ───────────────────────────────────────

    def test_round_trip_empty_graph(self, tmp_path: Path) -> None:
        """An empty graph should survive round-trip."""
        original = SkillGraph(capacity=42)
        save_path = tmp_path / "graph.json"
        original.save(save_path)

        restored = SkillGraph.load(save_path, capacity=42)
        assert len(restored) == 0
        assert restored._graph.number_of_edges() == 0

    def test_chinese_content(self, tmp_path: Path) -> None:
        """Chinese characters in policy/initiation_set should survive round-trip."""
        original = SkillGraph()
        original.add_skill(SkillNode(
            skill_id="sk-chinese",
            name="數學計算技能",
            policy="先分析問題，再逐步計算",
            initiation_set=["數學", "計算", "arithmetic"],
            termination="答案正確時終止",
        ))

        save_path = tmp_path / "graph.json"
        original.save(save_path)

        # Verify raw JSON is not ASCII-escaped
        raw = save_path.read_text(encoding="utf-8")
        assert "數學計算技能" in raw  # not \\u-escaped

        restored = SkillGraph.load(save_path)
        skill = restored.get_skill("sk-chinese")
        assert skill.name == "數學計算技能"
        assert "先分析問題" in skill.policy
        assert "數學" in skill.initiation_set
        assert skill.termination == "答案正確時終止"

    def test_multiple_save_load_cycles(self, tmp_path: Path) -> None:
        """Multiple save→load cycles should be idempotent."""
        save_path = tmp_path / "graph.json"

        g = SkillGraph()
        g.add_skill(SkillNode(skill_id="a", name="A"))
        g.save(save_path)

        for _ in range(3):
            g = SkillGraph.load(save_path)
            g.save(save_path)

        final = SkillGraph.load(save_path)
        assert len(final) == 1
        assert final.get_skill("a").name == "A"


# ═══════════════════════════════════════════════════════════════════
#  MemoryPartition persistence
# ═══════════════════════════════════════════════════════════════════


class TestMemoryPartitionPersistence:
    """Tests for MemoryPartition save/load functionality."""

    def test_to_dict_structure(self) -> None:
        """to_dict() should include thresholds and tier mapping."""
        mp = MemoryPartition(theta_high=0.8, theta_low=0.2)
        mp._tiers["a"] = "active"

        d = mp.to_dict()

        assert d["theta_high"] == 0.8
        assert d["theta_low"] == 0.2
        assert d["tiers"] == {"a": "active"}

    def test_from_dict_restores(self) -> None:
        """from_dict() should fully restore a partition."""
        data = {
            "theta_high": 0.75,
            "theta_low": 0.25,
            "epsilon_h": 0.05,
            "epsilon_l": 0.05,
            "tiers": {"x": "active", "y": "cold", "z": "archive"},
        }
        mp = MemoryPartition.from_dict(data)

        assert mp.theta_high == 0.75
        assert mp.theta_low == 0.25
        assert mp.epsilon_h == 0.05
        assert mp.get_tier("x") == "active"
        assert mp.get_tier("y") == "cold"
        assert mp.get_tier("z") == "archive"

    def test_save_load_tiers(self, tmp_path: Path) -> None:
        """Tier assignments should survive save/load."""
        partition = MemoryPartition(
            theta_high=0.7, theta_low=0.3,
            epsilon_h=0.05, epsilon_l=0.05,
        )
        partition._tiers["sk-001"] = "active"
        partition._tiers["sk-002"] = "cold"
        partition._tiers["sk-003"] = "archive"

        save_path = tmp_path / "partition.json"
        partition.save(save_path)

        restored = MemoryPartition.load(save_path)

        assert restored.get_tier("sk-001") == "active"
        assert restored.get_tier("sk-002") == "cold"
        assert restored.get_tier("sk-003") == "archive"
        assert restored.theta_high == 0.7
        assert restored.theta_low == 0.3
        assert restored.epsilon_h == 0.05
        assert restored.epsilon_l == 0.05

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """load() should return default partition if file missing."""
        mp = MemoryPartition.load(
            tmp_path / "missing.json",
            theta_high=0.9, theta_low=0.1,
        )
        assert mp.theta_high == 0.9
        assert mp.theta_low == 0.1
        assert len(mp._tiers) == 0

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        """load() should return default partition on bad JSON."""
        bad = tmp_path / "corrupt.json"
        bad.write_text("{{bad", encoding="utf-8")

        mp = MemoryPartition.load(bad)
        assert len(mp._tiers) == 0

    def test_untracked_skill_default_tier(self, tmp_path: Path) -> None:
        """Skills not in the partition should get default tier 'cold'."""
        partition = MemoryPartition()
        partition._tiers["known"] = "active"
        save_path = tmp_path / "p.json"
        partition.save(save_path)

        restored = MemoryPartition.load(save_path)
        assert restored.get_tier("known") == "active"
        assert restored.get_tier("unknown") == "cold"  # default

    def test_summary_counts_after_load(self, tmp_path: Path) -> None:
        """summary() should return correct counts after load."""
        mp = MemoryPartition()
        mp._tiers = {
            "a": "active", "b": "active",
            "c": "cold",
            "d": "archive", "e": "archive", "f": "archive",
        }
        save_path = tmp_path / "p.json"
        mp.save(save_path)

        restored = MemoryPartition.load(save_path)
        s = restored.summary()
        assert s == {"active": 2, "cold": 1, "archive": 3}
