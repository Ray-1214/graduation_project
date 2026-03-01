"""
Phase 1 Acceptance Tests — Self-Evolving Skill Graph.

18 test cases covering SkillNode, SkillGraph, and MemoryPartition.

Run with:
    pytest tests/test_skill_graph.py -v
"""

from __future__ import annotations

import math
import pytest

from skill_graph.skill_node import SkillNode
from skill_graph.skill_graph import SkillGraph
from skill_graph.memory_partition import MemoryPartition


# ── Helpers ──────────────────────────────────────────────────────────

def make_skill(name: str, utility: float = 0.0, **kw) -> SkillNode:
    sk = SkillNode(name=name, **kw)
    sk.utility = utility
    return sk


# =====================================================================
#  1.1 SkillNode Tests (1-1 ~ 1-5)
# =====================================================================

class TestSkillNode:

    def test_1_1_initialization_completeness(self):
        """SkillNode 建立後所有欄位皆有正確預設值"""
        sk = SkillNode()
        assert sk.skill_id.startswith("sk-")
        assert sk.policy == ""
        assert sk.termination == "Task is complete or max steps reached."
        assert sk.initiation_set == []
        assert sk.frequency == 0
        assert sk.reinforcement == 0.0
        assert sk.cost == 1.0
        assert sk.version == 1

    def test_1_2_compute_utility_formula(self):
        """α=1, β=1, γ_c=1, r=5, f=3, c=2 → U = 1×5 + 1×3 − 1×2 = 6.0"""
        sk = SkillNode(name="formula_test")
        sk.reinforcement = 5.0
        sk.frequency = 3
        sk.cost = 2.0
        u = sk.compute_utility(alpha=1.0, beta=1.0, gamma_c=1.0)
        assert u == 6.0
        assert sk.utility == 6.0

    def test_1_3_decay_correct(self):
        """utility=10, decay(0.1) → 9.0；連續 decay 兩次 → 8.1"""
        sk = make_skill("decay_test", utility=10.0)
        sk.decay(0.1)
        assert abs(sk.utility - 9.0) < 1e-9
        sk.decay(0.1)
        assert abs(sk.utility - 8.1) < 1e-9

    def test_1_4_decay_boundary(self):
        """γ=0 → utility 不變；γ=1 → utility 歸零"""
        sk1 = make_skill("no_decay", utility=5.0)
        sk1.decay(0.0)
        assert sk1.utility == 5.0

        sk2 = make_skill("full_decay", utility=5.0)
        sk2.decay(1.0)
        assert sk2.utility == 0.0

    def test_1_5_serialization_roundtrip(self):
        """to_dict() / from_dict() 可以 round-trip 而不丟失資料"""
        sk = SkillNode(
            name="serialize_me",
            policy="Use calculator for math",
            termination="Answer found",
            initiation_set=["math", "arithmetic"],
        )
        sk.frequency = 7
        sk.reinforcement = 3.5
        sk.cost = 1.2
        sk.version = 3
        sk.compute_utility(alpha=1.0, beta=0.5, gamma_c=0.1)

        d = sk.to_dict()
        sk2 = SkillNode.from_dict(d)

        assert sk2.skill_id == sk.skill_id
        assert sk2.name == sk.name
        assert sk2.policy == sk.policy
        assert sk2.termination == sk.termination
        assert sk2.initiation_set == sk.initiation_set
        assert sk2.frequency == sk.frequency
        assert sk2.reinforcement == sk.reinforcement
        assert sk2.cost == sk.cost
        assert sk2.version == sk.version
        assert abs(sk2.utility - sk.utility) < 1e-9


# =====================================================================
#  1.2 SkillGraph Tests (1-6 ~ 1-13)
# =====================================================================

class TestSkillGraph:

    def test_1_6_add_remove_skill(self):
        """add_skill 後 len +1；remove_skill 後 -1 且相關邊全部移除"""
        g = SkillGraph()
        a = make_skill("a")
        b = make_skill("b")
        c = make_skill("c")

        g.add_skill(a)
        assert len(g) == 1
        g.add_skill(b)
        assert len(g) == 2
        g.add_skill(c)
        assert len(g) == 3

        # Add edges to b
        g.add_edge(a.skill_id, b.skill_id, weight=1.0, edge_type="co_occurrence")
        g.add_edge(b.skill_id, c.skill_id, weight=1.0, edge_type="dependency")
        assert len(g.get_edges()) == 2

        # Remove b — should also remove both edges
        g.remove_skill(b.skill_id)
        assert len(g) == 2
        assert not g.has_skill(b.skill_id)
        assert len(g.get_edges()) == 0

    def test_1_7_edge_type_validation(self):
        """abstraction 邊形成環 → 拋出例外；co_occurrence 允許有環"""
        g = SkillGraph()
        a, b, c = make_skill("a"), make_skill("b"), make_skill("c")
        g.add_skill(a)
        g.add_skill(b)
        g.add_skill(c)

        # Abstraction chain: a → b → c
        g.add_edge(a.skill_id, b.skill_id, weight=1.0, edge_type="abstraction")
        g.add_edge(b.skill_id, c.skill_id, weight=1.0, edge_type="abstraction")

        # c → a would create cycle — must raise
        with pytest.raises(ValueError):
            g.add_edge(c.skill_id, a.skill_id, weight=1.0, edge_type="abstraction")

        # Rejected edge should NOT remain
        assert len(g.get_edges(edge_type="abstraction")) == 2

        # co_occurrence cycles are fine
        g.add_edge(a.skill_id, b.skill_id, weight=1.0, edge_type="co_occurrence")
        g.add_edge(b.skill_id, a.skill_id, weight=1.0, edge_type="co_occurrence")
        assert len(g.get_edges(edge_type="co_occurrence")) == 2

    def test_1_8_get_active_skills(self):
        """3 個 skill (U=0.8, 0.5, 0.2), threshold=0.4 → 回傳前 2 個"""
        g = SkillGraph()
        s1 = make_skill("high", utility=0.8)
        s2 = make_skill("mid", utility=0.5)
        s3 = make_skill("low", utility=0.2)
        g.add_skill(s1)
        g.add_skill(s2)
        g.add_skill(s3)

        active = g.get_active_skills(threshold=0.4)
        assert len(active) == 2
        active_names = {s.name for s in active}
        assert active_names == {"high", "mid"}

    def test_1_9_compute_entropy(self):
        """2 個 skill utility 相等 → H = ln(2) ≈ 0.693；1 個 skill → H = 0"""
        # Single skill → H = 0
        g1 = SkillGraph()
        g1.add_skill(make_skill("only", utility=1.0))
        assert abs(g1.compute_entropy()) < 1e-9

        # Two equal skills → H = ln(2)
        g2 = SkillGraph()
        g2.add_skill(make_skill("a", utility=1.0))
        g2.add_skill(make_skill("b", utility=1.0))
        expected = math.log(2)  # ≈ 0.6931
        assert abs(g2.compute_entropy() - expected) < 1e-6

    def test_1_10_compute_capacity(self):
        """5 個 skill, utility=[0.9,0.7,0.5,0.3,0.1], threshold=0.4 → capacity=3"""
        g = SkillGraph()
        for u in [0.9, 0.7, 0.5, 0.3, 0.1]:
            g.add_skill(make_skill(f"s{u}", utility=u))
        assert g.compute_capacity(threshold=0.4) == 3

    def test_1_11_decay_all(self):
        """全圖 decay 後每個 skill 的 utility 都乘以 (1−γ)"""
        g = SkillGraph()
        gamma = 0.1
        original_utils = [1.0, 2.0, 0.5, 3.0]
        skills = []
        for i, u in enumerate(original_utils):
            sk = make_skill(f"s{i}", utility=u)
            g.add_skill(sk)
            skills.append(sk)

        g.decay_all(gamma=gamma)

        for sk, orig in zip(skills, original_utils):
            expected = orig * (1.0 - gamma)
            assert abs(sk.utility - expected) < 1e-9, (
                f"{sk.name}: expected {expected}, got {sk.utility}"
            )

    def test_1_12_snapshot_completeness(self):
        """snapshot JSON 包含所有 nodes（含 metadata）和 edges（含 weight + type）"""
        g = SkillGraph(capacity=50)
        a = make_skill("alpha", utility=1.5)
        a.frequency = 3
        a.reinforcement = 2.0
        b = make_skill("beta", utility=0.8)
        g.add_skill(a)
        g.add_skill(b)
        g.add_edge(a.skill_id, b.skill_id, weight=0.7, edge_type="dependency")

        snap = g.snapshot()

        # Top-level fields
        assert snap["num_skills"] == 2
        assert snap["num_edges"] == 1
        assert snap["capacity"] == 50
        assert "structural_entropy" in snap
        assert "timestamp" in snap

        # Nodes contain skill metadata
        assert len(snap["nodes"]) == 2
        node_names = {n["name"] for n in snap["nodes"]}
        assert node_names == {"alpha", "beta"}
        alpha_node = next(n for n in snap["nodes"] if n["name"] == "alpha")
        assert alpha_node["frequency"] == 3
        assert alpha_node["reinforcement"] == 2.0
        assert alpha_node["utility"] == 1.5

        # Edges contain weight + type
        assert len(snap["edges"]) == 1
        edge = snap["edges"][0]
        assert edge["weight"] == 0.7
        assert edge["edge_type"] == "dependency"

    def test_1_13_empty_graph_safety(self):
        """空圖呼叫 compute_entropy(), decay_all(), snapshot() 不報錯"""
        g = SkillGraph()
        assert g.compute_entropy() == 0.0
        g.decay_all(gamma=0.1)  # no error
        snap = g.snapshot()
        assert snap["num_skills"] == 0
        assert snap["num_edges"] == 0
        assert snap["nodes"] == []
        assert snap["edges"] == []


# =====================================================================
#  1.3 MemoryPartition Tests (1-14 ~ 1-18)
# =====================================================================

class TestMemoryPartition:

    # θ_high=0.7, θ_low=0.3, ε_h=0.05, ε_l=0.05
    # Promote to active:    U ≥ 0.75  (0.7 + 0.05)
    # Demote from active:   U < 0.65  (0.7 - 0.05)
    # Promote from archive: U > 0.35  (0.3 + 0.05)
    # Demote to archive:    U ≤ 0.25  (0.3 - 0.05)

    MP_KWARGS = dict(theta_high=0.7, theta_low=0.3, epsilon_h=0.05, epsilon_l=0.05)

    def test_1_14_basic_partition(self):
        """U=0.8 → active, U=0.5 → cold, U=0.2 → archive"""
        mp = MemoryPartition(**self.MP_KWARGS)

        assert mp.assign_tier(make_skill("a", utility=0.8), "cold") == "active"
        assert mp.assign_tier(make_skill("c", utility=0.5), "cold") == "cold"
        assert mp.assign_tier(make_skill("r", utility=0.2), "cold") == "archive"

    def test_1_15_hysteresis_prevents_upgrade_oscillation(self):
        """
        skill 從 cold 升到 active (U=0.8)
        → utility 下降到 0.72 (仍 > θ_high − ε_h = 0.65)
        → 不降級，仍為 active
        """
        mp = MemoryPartition(**self.MP_KWARGS)
        sk = make_skill("osc", utility=0.8)

        # Step 1: cold → active
        tier = mp.assign_tier(sk, "cold")
        assert tier == "active"

        # Step 2: utility drops to 0.72 but above demotion threshold (0.65)
        sk.utility = 0.72
        tier = mp.assign_tier(sk, "active")
        assert tier == "active", (
            f"Hysteresis failed: U=0.72 > 0.65, should stay active, got {tier}"
        )

    def test_1_16_hysteresis_prevents_downgrade_oscillation(self):
        """
        skill 從 cold 降到 archive (U=0.2)
        → utility 上升到 0.28 (仍 < θ_low + ε_l = 0.35)
        → 不升級，仍為 archive
        """
        mp = MemoryPartition(**self.MP_KWARGS)
        sk = make_skill("osc", utility=0.2)

        # Step 1: cold → archive
        tier = mp.assign_tier(sk, "cold")
        assert tier == "archive"

        # Step 2: utility rises to 0.28 but below promotion threshold (0.35)
        sk.utility = 0.28
        tier = mp.assign_tier(sk, "archive")
        assert tier == "archive", (
            f"Hysteresis failed: U=0.28 < 0.35, should stay archive, got {tier}"
        )

    def test_1_17_update_all_batch(self):
        """圖中 10 個 skill，update_all 後回傳的 dict 包含所有 10 個 skill_id"""
        g = SkillGraph()
        skill_ids = []
        for i in range(10):
            sk = make_skill(f"s{i}", utility=i * 0.1)
            g.add_skill(sk)
            skill_ids.append(sk.skill_id)

        mp = MemoryPartition(**self.MP_KWARGS)
        result = mp.update_all(g)

        assert len(result) == 10
        for sid in skill_ids:
            assert sid in result
            assert result[sid] in ("active", "cold", "archive")

    def test_1_18_cross_stage_consistency(self):
        """MemoryPartition 的 tier 資訊和 SkillGraph snapshot 中節點的 tier 欄位一致"""
        g = SkillGraph()
        skills = [
            make_skill("act1", utility=0.9),
            make_skill("act2", utility=0.8),
            make_skill("cold1", utility=0.5),
            make_skill("cold2", utility=0.4),
            make_skill("arc1", utility=0.1),
        ]
        for sk in skills:
            g.add_skill(sk)

        mp = MemoryPartition(**self.MP_KWARGS)
        tier_result = mp.update_all(g)

        # Get snapshot WITH partition
        snap = g.snapshot(partition=mp)

        # Every node in snapshot must have a tier field matching partition
        for node_dict in snap["nodes"]:
            sid = node_dict["skill_id"]
            assert "tier" in node_dict, f"Node {sid} missing 'tier' field in snapshot"
            assert node_dict["tier"] == tier_result[sid], (
                f"Node {sid}: snapshot tier={node_dict['tier']} != "
                f"partition tier={tier_result[sid]}"
            )
