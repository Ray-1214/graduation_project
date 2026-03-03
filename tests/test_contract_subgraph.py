"""
Tests for contract_subgraph — Macro-Skill Composition.

Covers:
  - Basic contraction: 3 skills → 1 macro-skill
  - Edge rewiring: in-edges and out-edges correctly redirected
  - Policy/termination/initiation composition
  - Metadata: cost=sum, reinforcement=average
  - Reachability preservation after contraction
  - Internal edges between sequence members are removed
  - Error cases: too few skills, missing nodes, duplicates, ID conflict
  - Contraction at graph boundary (head / tail nodes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode
from skill_graph.contract_subgraph import contract_subgraph


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_skill(sid: str, **kwargs) -> SkillNode:
    defaults = dict(
        skill_id=sid,
        name=f"skill_{sid}",
        policy=f"do_{sid}",
        termination=f"done_{sid}",
        initiation_set=[f"tag_{sid}"],
        cost=1.0,
        reinforcement=2.0,
    )
    defaults.update(kwargs)
    return SkillNode(**defaults)


def _build_chain_graph() -> SkillGraph:
    """Build a linear chain:  A → B → C → D → E

    With co_occurrence edges along the chain.
    """
    g = SkillGraph(capacity=20)
    for sid in ["A", "B", "C", "D", "E"]:
        g.add_skill(_make_skill(sid))

    g.add_edge("A", "B", weight=1.0, edge_type="co_occurrence")
    g.add_edge("B", "C", weight=2.0, edge_type="co_occurrence")
    g.add_edge("C", "D", weight=3.0, edge_type="co_occurrence")
    g.add_edge("D", "E", weight=1.5, edge_type="co_occurrence")
    return g


# ═══════════════════════════════════════════════════════════════════
#  Test Cases
# ═══════════════════════════════════════════════════════════════════

class TestContractSubgraph:

    # ── Basic contraction ────────────────────────────────────────

    def test_basic_contraction(self):
        """Contract B,C,D into macro M in chain A→B→C→D→E."""
        g = _build_chain_graph()
        assert len(g) == 5

        macro = contract_subgraph(g, ["B", "C", "D"], "M")

        # Graph should now have 3 nodes: A, M, E
        assert len(g) == 3
        assert g.has_skill("M")
        assert not g.has_skill("B")
        assert not g.has_skill("C")
        assert not g.has_skill("D")

    # ── Edge rewiring ────────────────────────────────────────────

    def test_edge_rewiring(self):
        """In-edges to first and out-edges from last are rewired."""
        g = _build_chain_graph()
        macro = contract_subgraph(g, ["B", "C", "D"], "M")

        edges = g.get_edges()
        edge_pairs = [(e["src"], e["dst"]) for e in edges]

        # A → M (was A → B)
        assert ("A", "M") in edge_pairs
        # M → E (was D → E)
        assert ("M", "E") in edge_pairs
        # No edges to/from old nodes
        for e in edges:
            assert e["src"] not in {"B", "C", "D"}
            assert e["dst"] not in {"B", "C", "D"}

    def test_edge_weight_preserved(self):
        """Rewired edges keep their original weights."""
        g = _build_chain_graph()
        contract_subgraph(g, ["B", "C", "D"], "M")

        edges = {(e["src"], e["dst"]): e for e in g.get_edges()}
        # A→B had weight 1.0 → A→M should be 1.0
        assert edges[("A", "M")]["weight"] == 1.0
        # D→E had weight 1.5 → M→E should be 1.5
        assert edges[("M", "E")]["weight"] == 1.5

    # ── Policy / termination / initiation composition ────────────

    def test_policy_composition(self):
        """Macro policy = first → second → … → last."""
        g = _build_chain_graph()
        macro = contract_subgraph(g, ["B", "C", "D"], "M")

        assert macro.policy == "do_B → do_C → do_D"

    def test_termination_from_last(self):
        """Macro termination = last skill's termination."""
        g = _build_chain_graph()
        macro = contract_subgraph(g, ["B", "C", "D"], "M")

        assert macro.termination == "done_D"

    def test_initiation_from_first(self):
        """Macro initiation set = first skill's initiation set."""
        g = _build_chain_graph()
        macro = contract_subgraph(g, ["B", "C", "D"], "M")

        assert macro.initiation_set == ["tag_B"]

    # ── Metadata ─────────────────────────────────────────────────

    def test_cost_is_sum(self):
        """Macro cost = sum of individual costs."""
        g = _build_chain_graph()
        # Each skill has cost=1.0, 3 skills → cost=3.0
        macro = contract_subgraph(g, ["B", "C", "D"], "M")

        assert macro.cost == 3.0

    def test_reinforcement_is_average(self):
        """Macro reinforcement = average of individual reinforcements."""
        g = _build_chain_graph()
        # Each skill has reinforcement=2.0, 3 skills → avg=2.0
        macro = contract_subgraph(g, ["B", "C", "D"], "M")

        assert macro.reinforcement == 2.0

    def test_frequency_reset(self):
        """Macro frequency starts at 0."""
        g = _build_chain_graph()
        macro = contract_subgraph(g, ["B", "C", "D"], "M")

        assert macro.frequency == 0

    def test_tags_include_macro(self):
        """Macro has 'macro' and 'contracted' tags."""
        g = _build_chain_graph()
        macro = contract_subgraph(g, ["B", "C", "D"], "M")

        assert "macro" in macro.tags
        assert "contracted" in macro.tags

    # ── Reachability preservation ────────────────────────────────

    def test_reachability_preserved(self):
        """A can still reach E after contracting B,C,D."""
        import networkx as nx

        g = _build_chain_graph()
        # Before: A → B → C → D → E
        # A can reach E
        assert nx.has_path(g._graph, "A", "E")

        contract_subgraph(g, ["B", "C", "D"], "M")
        # After: A → M → E
        assert nx.has_path(g._graph, "A", "E")

    # ── Internal edges removed ───────────────────────────────────

    def test_internal_edges_removed(self):
        """Edges between sequence members (B→C, C→D) are gone."""
        g = _build_chain_graph()
        contract_subgraph(g, ["B", "C", "D"], "M")

        for e in g.get_edges():
            assert e["src"] not in {"B", "C", "D"}
            assert e["dst"] not in {"B", "C", "D"}

    # ── Contraction at boundaries ────────────────────────────────

    def test_contract_head(self):
        """Contract the first two nodes (A,B) — no in-edges to rewire."""
        g = _build_chain_graph()
        macro = contract_subgraph(g, ["A", "B"], "HEAD")

        assert len(g) == 4  # HEAD, C, D, E
        assert g.has_skill("HEAD")
        edges = [(e["src"], e["dst"]) for e in g.get_edges()]
        assert ("HEAD", "C") in edges

    def test_contract_tail(self):
        """Contract the last two nodes (D,E) — no out-edges to rewire."""
        g = _build_chain_graph()
        macro = contract_subgraph(g, ["D", "E"], "TAIL")

        assert len(g) == 4  # A, B, C, TAIL
        assert g.has_skill("TAIL")
        edges = [(e["src"], e["dst"]) for e in g.get_edges()]
        assert ("C", "TAIL") in edges

    def test_contract_pair(self):
        """Contract exactly 2 nodes — minimum valid sequence."""
        g = _build_chain_graph()
        macro = contract_subgraph(g, ["C", "D"], "CD")

        assert len(g) == 4  # A, B, CD, E
        assert macro.policy == "do_C → do_D"
        edges = [(e["src"], e["dst"]) for e in g.get_edges()]
        assert ("B", "CD") in edges
        assert ("CD", "E") in edges

    # ── Multiple external edges ──────────────────────────────────

    def test_multiple_in_edges(self):
        """Multiple nodes pointing to the first skill → all rewired."""
        g = SkillGraph(capacity=20)
        for sid in ["X1", "X2", "A", "B", "C"]:
            g.add_skill(_make_skill(sid))
        g.add_edge("X1", "A", weight=1.0)
        g.add_edge("X2", "A", weight=2.0)
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=1.0)

        contract_subgraph(g, ["A", "B"], "AB")

        edges = [(e["src"], e["dst"]) for e in g.get_edges()]
        assert ("X1", "AB") in edges
        assert ("X2", "AB") in edges
        assert ("AB", "C") in edges

    # ── Error cases ──────────────────────────────────────────────

    def test_error_too_few_skills(self):
        """Sequence with < 2 skills raises ValueError."""
        g = _build_chain_graph()
        with pytest.raises(ValueError, match="at least 2"):
            contract_subgraph(g, ["A"], "M")

    def test_error_missing_skill(self):
        """Missing skill_id in graph raises ValueError."""
        g = _build_chain_graph()
        with pytest.raises(ValueError, match="not found"):
            contract_subgraph(g, ["A", "MISSING"], "M")

    def test_error_duplicate_in_sequence(self):
        """Duplicate skill_id in sequence raises ValueError."""
        g = _build_chain_graph()
        with pytest.raises(ValueError, match="duplicates"):
            contract_subgraph(g, ["A", "B", "A"], "M")

    def test_error_id_already_exists(self):
        """New skill ID already in graph raises ValueError."""
        g = _build_chain_graph()
        with pytest.raises(ValueError, match="already exists"):
            contract_subgraph(g, ["B", "C"], "A")  # A already exists
