"""
Pytest tests for the Self-Evolving Skill Graph system.

Covers: SkillNode, SkillGraph, MemoryPartition.

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
#  SkillNode
# =====================================================================

class TestSkillNode:

    def test_creation_and_defaults(self):
        sk = SkillNode(name="test")
        assert sk.name == "test"
        assert sk.frequency == 0
        assert sk.reinforcement == 0.0
        assert sk.version == 1
        assert sk.skill_id.startswith("sk-")

    def test_compute_utility(self):
        sk = SkillNode(name="calc", frequency=4, reinforcement=2.0, cost=1.0)
        u = sk.compute_utility(alpha=1.0, beta=0.5, gamma_c=0.1)
        # U = 1.0*2.0 + 0.5*4 - 0.1*1.0 = 3.9
        assert abs(u - 3.9) < 1e-9
        assert abs(sk.utility - 3.9) < 1e-9

    def test_decay(self):
        sk = make_skill("d", utility=1.0)
        sk.decay(gamma=0.1)
        assert abs(sk.utility - 0.9) < 1e-9

    def test_reinforce(self):
        sk = SkillNode(name="r")
        sk.reinforce(delta_u=0.8, cost=2.0)
        assert sk.frequency == 1
        assert sk.reinforcement == 0.8
        assert sk.utility == 0.8

    def test_matches(self):
        sk = SkillNode(name="m", initiation_set=["math", "calculate"])
        assert sk.matches("calculate 2+3")
        assert sk.matches("MATH quiz")
        assert not sk.matches("write a poem")

    def test_evolve(self):
        parent = make_skill("parent", utility=5.0)
        parent.frequency = 10
        child = parent.evolve(new_policy="improved policy")
        assert child.parent_id == parent.skill_id
        assert child.version == parent.version + 1
        assert child.frequency == 0
        assert child.utility == 0.0
        assert child.policy == "improved policy"

    def test_serialization_roundtrip(self):
        sk = SkillNode(name="ser", policy="do X", initiation_set=["a", "b"])
        sk.reinforce(0.5, 1.0)
        d = sk.to_dict()
        sk2 = SkillNode.from_dict(d)
        assert sk2.skill_id == sk.skill_id
        assert sk2.name == sk.name
        assert sk2.reinforcement == sk.reinforcement


# =====================================================================
#  SkillGraph
# =====================================================================

class TestSkillGraph:

    # ── Nodes ────────────────────────────────────────────────────────

    def test_add_and_get_skill(self):
        g = SkillGraph()
        sk = make_skill("alpha", utility=1.0)
        g.add_skill(sk)
        assert g.has_skill(sk.skill_id)
        assert g.get_skill(sk.skill_id) is sk
        assert len(g) == 1

    def test_add_duplicate_raises(self):
        g = SkillGraph()
        sk = make_skill("dup")
        g.add_skill(sk)
        with pytest.raises(ValueError):
            g.add_skill(sk)

    def test_capacity_enforced(self):
        g = SkillGraph(capacity=2)
        g.add_skill(make_skill("a"))
        g.add_skill(make_skill("b"))
        with pytest.raises(OverflowError):
            g.add_skill(make_skill("c"))

    def test_remove_skill_and_edges(self):
        g = SkillGraph()
        a, b = make_skill("a"), make_skill("b")
        g.add_skill(a)
        g.add_skill(b)
        g.add_edge(a.skill_id, b.skill_id, weight=1.0)
        g.remove_skill(a.skill_id)
        assert not g.has_skill(a.skill_id)
        assert len(g.get_edges()) == 0

    # ── Edges ────────────────────────────────────────────────────────

    def test_add_edge_all_types(self):
        g = SkillGraph()
        a, b, c = make_skill("a"), make_skill("b"), make_skill("c")
        g.add_skill(a); g.add_skill(b); g.add_skill(c)
        g.add_edge(a.skill_id, b.skill_id, weight=0.8, edge_type="co_occurrence")
        g.add_edge(a.skill_id, c.skill_id, weight=1.0, edge_type="dependency")
        g.add_edge(b.skill_id, c.skill_id, weight=0.5, edge_type="abstraction")
        assert len(g.get_edges()) == 3
        assert len(g.get_edges(edge_type="abstraction")) == 1

    def test_add_edge_rejects_non_positive_weight(self):
        g = SkillGraph()
        a, b = make_skill("a"), make_skill("b")
        g.add_skill(a); g.add_skill(b)
        with pytest.raises(ValueError):
            g.add_edge(a.skill_id, b.skill_id, weight=0.0)

    def test_add_edge_rejects_missing_node(self):
        g = SkillGraph()
        a = make_skill("a")
        g.add_skill(a)
        with pytest.raises(KeyError):
            g.add_edge(a.skill_id, "ghost", weight=1.0)

    def test_abstraction_dag_enforced(self):
        g = SkillGraph()
        a, b, c = make_skill("a"), make_skill("b"), make_skill("c")
        g.add_skill(a); g.add_skill(b); g.add_skill(c)
        g.add_edge(a.skill_id, b.skill_id, weight=1.0, edge_type="abstraction")
        g.add_edge(b.skill_id, c.skill_id, weight=1.0, edge_type="abstraction")
        with pytest.raises(ValueError, match="[Cc]ycle|DAG"):
            g.add_edge(c.skill_id, a.skill_id, weight=1.0, edge_type="abstraction")
        # Rejected edge should NOT remain
        assert len(g.get_edges(edge_type="abstraction")) == 2

    def test_co_occurrence_allows_cycles(self):
        g = SkillGraph()
        a, b = make_skill("a"), make_skill("b")
        g.add_skill(a); g.add_skill(b)
        g.add_edge(a.skill_id, b.skill_id, weight=1.0, edge_type="co_occurrence")
        g.add_edge(b.skill_id, a.skill_id, weight=1.0, edge_type="co_occurrence")
        assert len(g.get_edges()) == 2

    # ── Queries ──────────────────────────────────────────────────────

    def test_get_active_skills(self):
        g = SkillGraph()
        g.add_skill(make_skill("high", utility=0.8))
        g.add_skill(make_skill("low", utility=0.2))
        g.add_skill(make_skill("zero", utility=0.0))
        active = g.get_active_skills(threshold=0.5)
        assert len(active) == 1
        assert active[0].name == "high"

    def test_get_matching_skills(self):
        g = SkillGraph()
        g.add_skill(make_skill("calc", initiation_set=["math", "calculate"]))
        g.add_skill(make_skill("write", initiation_set=["essay"]))
        assert len(g.get_matching_skills("calculate 2+3")) == 1

    # ── Structural entropy ───────────────────────────────────────────

    def test_entropy_empty_graph(self):
        assert SkillGraph().compute_entropy() == 0.0

    def test_entropy_single_skill(self):
        g = SkillGraph()
        g.add_skill(make_skill("only", utility=1.0))
        assert abs(g.compute_entropy()) < 1e-9

    def test_entropy_uniform(self):
        g = SkillGraph()
        n = 4
        for i in range(n):
            g.add_skill(make_skill(f"s{i}", utility=1.0))
        assert abs(g.compute_entropy() - math.log2(n)) < 1e-9

    def test_entropy_skewed_less_than_max(self):
        g = SkillGraph()
        g.add_skill(make_skill("dominant", utility=10.0))
        g.add_skill(make_skill("weak", utility=0.1))
        h = g.compute_entropy()
        assert 0 < h < math.log2(2)

    # ── Capacity ─────────────────────────────────────────────────────

    def test_compute_capacity(self):
        g = SkillGraph()
        g.add_skill(make_skill("a", utility=0.9))
        g.add_skill(make_skill("b", utility=0.6))
        g.add_skill(make_skill("c", utility=0.3))
        assert g.compute_capacity(threshold=0.5) == 2

    # ── Decay ────────────────────────────────────────────────────────

    def test_decay_all(self):
        g = SkillGraph()
        a = make_skill("a", utility=1.0)
        b = make_skill("b", utility=2.0)
        g.add_skill(a); g.add_skill(b)
        g.decay_all(gamma=0.1)
        assert abs(a.utility - 0.9) < 1e-9
        assert abs(b.utility - 1.8) < 1e-9

    # ── Subgraph & snapshot ──────────────────────────────────────────

    def test_get_subgraph(self):
        g = SkillGraph()
        a, b, c = make_skill("a"), make_skill("b"), make_skill("c")
        g.add_skill(a); g.add_skill(b); g.add_skill(c)
        g.add_edge(a.skill_id, b.skill_id, weight=1.0)
        g.add_edge(b.skill_id, c.skill_id, weight=1.0)
        sub = g.get_subgraph([a.skill_id, b.skill_id])
        assert len(sub) == 2
        assert sub.number_of_edges() == 1

    def test_snapshot(self):
        g = SkillGraph(capacity=50)
        a = make_skill("a", utility=1.0)
        b = make_skill("b", utility=2.0)
        g.add_skill(a); g.add_skill(b)
        g.add_edge(a.skill_id, b.skill_id, weight=0.5, edge_type="dependency")
        snap = g.snapshot()
        assert snap["num_skills"] == 2
        assert snap["num_edges"] == 1
        assert snap["capacity"] == 50
        assert "structural_entropy" in snap
        assert snap["edges"][0]["edge_type"] == "dependency"


# =====================================================================
#  MemoryPartition
# =====================================================================

class TestMemoryPartition:

    # θ_high=0.7, θ_low=0.3, ε_h=0.1, ε_l=0.1
    # Promote to active:    U ≥ 0.8
    # Demote from active:   U < 0.6
    # Promote from archive: U > 0.4
    # Demote to archive:    U ≤ ~0.2

    # ── Basic assignment ─────────────────────────────────────────────

    def test_cold_high_utility_to_active(self):
        mp = MemoryPartition()
        assert mp.assign_tier(make_skill("h", utility=0.9), "cold") == "active"

    def test_cold_mid_utility_stays_cold(self):
        mp = MemoryPartition()
        assert mp.assign_tier(make_skill("m", utility=0.5), "cold") == "cold"

    def test_cold_low_utility_to_archive(self):
        mp = MemoryPartition()
        assert mp.assign_tier(make_skill("l", utility=0.1), "cold") == "archive"

    def test_archive_high_utility_jumps_to_active(self):
        mp = MemoryPartition()
        assert mp.assign_tier(make_skill("r", utility=0.85), "archive") == "active"

    def test_active_very_low_skips_to_archive(self):
        mp = MemoryPartition()
        assert mp.assign_tier(make_skill("c", utility=0.1), "active") == "archive"

    # ── Hysteresis: active boundary ──────────────────────────────────

    def test_active_stays_in_hysteresis_band(self):
        mp = MemoryPartition()
        # U=0.65 is in dead band [0.6, 0.8): active stays, cold stays
        assert mp.assign_tier(make_skill("x", utility=0.65), "active") == "active"
        assert mp.assign_tier(make_skill("x", utility=0.65), "cold") == "cold"

    def test_active_demotes_below_hysteresis(self):
        mp = MemoryPartition()
        assert mp.assign_tier(make_skill("x", utility=0.55), "active") == "cold"

    # ── Hysteresis: archive boundary ─────────────────────────────────

    def test_archive_stays_in_hysteresis_band(self):
        mp = MemoryPartition()
        assert mp.assign_tier(make_skill("x", utility=0.35), "archive") == "archive"

    def test_archive_promotes_above_hysteresis(self):
        mp = MemoryPartition()
        assert mp.assign_tier(make_skill("x", utility=0.45), "archive") == "cold"

    # ── Anti-oscillation (full trajectory) ───────────────────────────

    def test_no_oscillation_at_active_boundary(self):
        """6 episodes around θ_high — hysteresis prevents flip-flop."""
        mp = MemoryPartition()
        sk = make_skill("osc")
        trajectory = [
            (0.85, "active"),   # promote
            (0.72, "active"),   # hysteresis holds
            (0.65, "active"),   # still holds
            (0.58, "cold"),     # drops below 0.6
            (0.72, "cold"),     # can't re-promote (< 0.8)
            (0.82, "active"),   # crosses 0.8 again
        ]
        tier = "cold"
        for i, (u, expected) in enumerate(trajectory):
            sk.utility = u
            tier = mp.assign_tier(sk, tier)
            assert tier == expected, (
                f"Episode {i+1}: U={u}, expected {expected}, got {tier}"
            )

    def test_no_oscillation_at_archive_boundary(self):
        """6 episodes around θ_low — hysteresis prevents flip-flop."""
        mp = MemoryPartition()
        sk = make_skill("osc2")
        trajectory = [
            (0.10, "archive"),
            (0.25, "archive"),  # holds
            (0.38, "archive"),  # still under 0.4
            (0.45, "cold"),     # crosses 0.4
            (0.35, "cold"),     # can't re-demote (> 0.2)
            (0.15, "archive"),  # drops below 0.2
        ]
        tier = "cold"
        for i, (u, expected) in enumerate(trajectory):
            sk.utility = u
            tier = mp.assign_tier(sk, tier)
            assert tier == expected, (
                f"Episode {i+1}: U={u}, expected {expected}, got {tier}"
            )

    # ── Bulk update with SkillGraph ──────────────────────────────────

    def test_update_all_partitions_correctly(self):
        g = SkillGraph()
        a = make_skill("act", utility=0.9)
        b = make_skill("cld", utility=0.5)
        c = make_skill("arc", utility=0.1)
        g.add_skill(a); g.add_skill(b); g.add_skill(c)

        mp = MemoryPartition()
        result = mp.update_all(g)
        assert result[a.skill_id] == "active"
        assert result[b.skill_id] == "cold"
        assert result[c.skill_id] == "archive"

    def test_update_all_preserves_hysteresis(self):
        g = SkillGraph()
        sk = make_skill("trk", utility=0.85)
        g.add_skill(sk)

        mp = MemoryPartition()
        mp.update_all(g)
        assert mp.get_tier(sk.skill_id) == "active"

        sk.utility = 0.65
        mp.update_all(g)
        assert mp.get_tier(sk.skill_id) == "active"  # hysteresis

        sk.utility = 0.55
        mp.update_all(g)
        assert mp.get_tier(sk.skill_id) == "cold"

    def test_summary_counts(self):
        g = SkillGraph()
        g.add_skill(make_skill("a1", utility=0.9))
        g.add_skill(make_skill("a2", utility=0.85))
        g.add_skill(make_skill("c1", utility=0.5))
        g.add_skill(make_skill("ar", utility=0.1))
        mp = MemoryPartition()
        mp.update_all(g)
        s = mp.summary()
        assert s == {"active": 2, "cold": 1, "archive": 1}

    # ── Edge cases ───────────────────────────────────────────────────

    def test_invalid_thresholds_raises(self):
        with pytest.raises(ValueError):
            MemoryPartition(theta_high=0.3, theta_low=0.7)

    def test_untracked_skill_defaults_to_cold(self):
        mp = MemoryPartition()
        assert mp.get_tier("nonexistent") == "cold"
