"""
SkillGraph — weighted directed graph of reusable skills.

Formal definition:  G_t = (Σ_t, E_t, W_t)  where
  Σ_t  = set of SkillNode vertices
  E_t  ⊆ Σ_t × Σ_t  = directed edges
  W_t  : E_t → ℝ>0   = positive edge weights

Edge semantic types:
  (a) co_occurrence   — sequential co-occurrence in a trace (cycles OK)
  (b) dependency      — functional prerequisite (σ_i must run before σ_j)
  (c) abstraction     — macro-skill composition (MUST form a DAG)

Structural entropy:
  H(G_t) = −Σ p(σ) log p(σ),  p(σ) = U(σ) / ΣU

Structural capacity:
  K* = max{|Σ| : ∀σ, U(σ) ≥ θ}   →   enforced |Σ_t| ≤ K
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set

import networkx as nx  # type: ignore

from skill_graph.skill_node import SkillNode

EdgeType = Literal["co_occurrence", "dependency", "abstraction"]


class SkillGraph:
    """Weighted directed graph of SkillNode instances.

    Internally backed by a ``networkx.DiGraph``.  Each node stores
    its ``SkillNode`` object as attribute ``"skill"``.  Each edge
    stores ``weight: float`` and ``edge_type: EdgeType``.
    """

    def __init__(self, capacity: int = 100) -> None:
        """
        Args:
            capacity: Upper-bound K on the number of skills |Σ_t| ≤ K.
        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self.capacity: int = capacity

    # ── Node operations ──────────────────────────────────────────────

    def add_skill(self, skill: SkillNode) -> None:
        """Add a skill node to the graph.

        Raises:
            ValueError: If ``skill.skill_id`` already exists.
            OverflowError: If adding would exceed capacity K.
        """
        if skill.skill_id in self._graph:
            raise ValueError(f"Skill '{skill.skill_id}' already exists in graph.")
        if len(self._graph) >= self.capacity:
            raise OverflowError(
                f"Graph at capacity ({self.capacity}). "
                f"Remove a skill or increase capacity before adding."
            )
        self._graph.add_node(skill.skill_id, skill=skill)

    def remove_skill(self, skill_id: str) -> None:
        """Remove a skill node and all incident edges.

        Raises:
            KeyError: If ``skill_id`` is not in the graph.
        """
        if skill_id not in self._graph:
            raise KeyError(f"Skill '{skill_id}' not found in graph.")
        self._graph.remove_node(skill_id)

    def get_skill(self, skill_id: str) -> SkillNode:
        """Retrieve a SkillNode by ID.

        Raises:
            KeyError: If ``skill_id`` is not in the graph.
        """
        if skill_id not in self._graph:
            raise KeyError(f"Skill '{skill_id}' not found in graph.")
        return self._graph.nodes[skill_id]["skill"]

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self._graph

    # ── Edge operations ──────────────────────────────────────────────

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        weight: float = 1.0,
        edge_type: EdgeType = "co_occurrence",
    ) -> None:
        """Add a directed edge with weight and semantic type.

        Args:
            src_id:    Source skill ID.
            dst_id:    Destination skill ID.
            weight:    Positive edge weight.
            edge_type: One of ``"co_occurrence"``, ``"dependency"``,
                       ``"abstraction"``.

        Raises:
            KeyError: If either node is missing.
            ValueError: If weight ≤ 0.
            ValueError: If adding an ``abstraction`` edge would break
                        the DAG invariant.
        """
        if src_id not in self._graph:
            raise KeyError(f"Source skill '{src_id}' not in graph.")
        if dst_id not in self._graph:
            raise KeyError(f"Destination skill '{dst_id}' not in graph.")
        if weight <= 0:
            raise ValueError(f"Edge weight must be positive, got {weight}.")

        # --- DAG invariant for abstraction edges ---
        if edge_type == "abstraction":
            # Temporarily add the edge and check for cycles
            self._graph.add_edge(
                src_id, dst_id, weight=weight, edge_type=edge_type,
            )
            abstraction_subgraph = self._get_typed_subgraph("abstraction")
            if not nx.is_directed_acyclic_graph(abstraction_subgraph):
                self._graph.remove_edge(src_id, dst_id)
                raise ValueError(
                    f"Adding abstraction edge {src_id} → {dst_id} "
                    f"would create a cycle. Abstraction edges must form a DAG."
                )
            # Edge already added above, done
            return

        self._graph.add_edge(
            src_id, dst_id, weight=weight, edge_type=edge_type,
        )

    def remove_edge(self, src_id: str, dst_id: str) -> None:
        """Remove a directed edge."""
        self._graph.remove_edge(src_id, dst_id)

    def get_edges(
        self,
        edge_type: Optional[EdgeType] = None,
    ) -> List[Dict[str, Any]]:
        """List edges, optionally filtered by type."""
        result = []
        for u, v, data in self._graph.edges(data=True):
            if edge_type is None or data.get("edge_type") == edge_type:
                result.append({
                    "src": u,
                    "dst": v,
                    "weight": data.get("weight", 1.0),
                    "edge_type": data.get("edge_type", "co_occurrence"),
                })
        return result

    # ── Queries ──────────────────────────────────────────────────────

    def get_active_skills(self, threshold: float = 0.0) -> List[SkillNode]:
        """Return all skills with ``utility ≥ threshold``."""
        return [
            data["skill"]
            for _, data in self._graph.nodes(data=True)
            if data["skill"].utility >= threshold
        ]

    def get_matching_skills(self, task: str) -> List[SkillNode]:
        """Return skills whose initiation set matches the task."""
        return [
            data["skill"]
            for _, data in self._graph.nodes(data=True)
            if data["skill"].matches(task)
        ]

    # ── Structural entropy ───────────────────────────────────────────

    def compute_entropy(self) -> float:
        """Compute structural entropy H(G_t).

        H(G_t) = −Σ_{σ∈Σ} p(σ) log₂ p(σ)

        where p(σ) = U(σ) / Σ U(σ')  (normalised utility).

        Skills with U ≤ 0 are excluded.  Returns 0.0 for an empty
        graph or when all utilities are ≤ 0.
        """
        utilities = [
            data["skill"].utility
            for _, data in self._graph.nodes(data=True)
            if data["skill"].utility > 0
        ]
        if not utilities:
            return 0.0

        total = sum(utilities)
        entropy = 0.0
        for u in utilities:
            p = u / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    # ── Structural capacity ──────────────────────────────────────────

    def compute_capacity(self, threshold: float = 0.0) -> int:
        """Compute structural capacity K*.

        K* = |{σ ∈ Σ : U(σ) ≥ θ}|
        """
        return sum(
            1
            for _, data in self._graph.nodes(data=True)
            if data["skill"].utility >= threshold
        )

    # ── Decay ────────────────────────────────────────────────────────

    def decay_all(self, gamma: float = 0.05) -> None:
        """Apply multiplicative utility decay to every skill.

        U_{t+1}(σ) = (1 − γ) · U_t(σ)

        Args:
            gamma: Decay rate γ ∈ (0, 1).
        """
        for _, data in self._graph.nodes(data=True):
            data["skill"].decay(gamma)

    # ── Subgraph ─────────────────────────────────────────────────────

    def get_subgraph(self, skill_ids: List[str]) -> nx.DiGraph:
        """Return an induced subgraph (view) over the given skill IDs."""
        present = [sid for sid in skill_ids if sid in self._graph]
        return self._graph.subgraph(present).copy()

    def _get_typed_subgraph(self, edge_type: EdgeType) -> nx.DiGraph:
        """Return a subgraph containing only edges of the given type."""
        g = nx.DiGraph()
        g.add_nodes_from(self._graph.nodes(data=True))
        for u, v, data in self._graph.edges(data=True):
            if data.get("edge_type") == edge_type:
                g.add_edge(u, v, **data)
        return g

    # ── Snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Export the full graph state as a serialisable dict."""
        nodes = []
        for nid, data in self._graph.nodes(data=True):
            nodes.append(data["skill"].to_dict())

        edges = []
        for u, v, data in self._graph.edges(data=True):
            edges.append({
                "src": u,
                "dst": v,
                "weight": data.get("weight", 1.0),
                "edge_type": data.get("edge_type", "co_occurrence"),
            })

        active = self.get_active_skills(threshold=0.0)
        return {
            "timestamp": time.time(),
            "num_skills": len(self._graph),
            "num_edges": self._graph.number_of_edges(),
            "capacity": self.capacity,
            "structural_entropy": round(self.compute_entropy(), 4),
            "nodes": nodes,
            "edges": edges,
        }

    # ── Dunder ───────────────────────────────────────────────────────

    @property
    def skills(self) -> List[SkillNode]:
        """All skill nodes in the graph."""
        return [d["skill"] for _, d in self._graph.nodes(data=True)]

    def __len__(self) -> int:
        return len(self._graph)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._graph

    def __repr__(self) -> str:
        return (
            f"SkillGraph(skills={len(self._graph)}, "
            f"edges={self._graph.number_of_edges()}, "
            f"cap={self.capacity})"
        )
