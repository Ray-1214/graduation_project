"""
contract_subgraph — Macro-Skill Composition via Subgraph Contraction.

Given a skill subsequence (σᵢ, σᵢ₊₁, …, σⱼ) that co-occurs with empirical
probability P > δ, contract it into a single macro-skill σ*:

    π_σ* = π_σᵢ ∘ π_σᵢ₊₁ ∘ … ∘ π_σⱼ   (composed policy)
    β_σ* = β_σⱼ                          (last skill's termination)
    I_σ* = I_σᵢ                          (first skill's initiation set)

Graph operation:
  1. Create macro-skill node σ*.
  2. Rewire all in-edges to σᵢ → point to σ*.
  3. Rewire all out-edges from σⱼ → originate from σ*.
  4. Remove the original subsequence nodes and internal edges.
  5. Add σ* with rewired edges.

Invariant: reachability is preserved — any node reachable before
contraction is still reachable after.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from skill_graph.skill_graph import SkillGraph, EdgeType
from skill_graph.skill_node import SkillNode

logger = logging.getLogger(__name__)


def contract_subgraph(
    graph: SkillGraph,
    skill_sequence: List[str],
    new_skill_id: str,
    name: Optional[str] = None,
) -> SkillNode:
    """Contract a skill subsequence into a single macro-skill node.

    Args:
        graph:          The SkillGraph to modify **in-place**.
        skill_sequence: Ordered list of skill_ids to merge
                        (e.g. ``["s1", "s2", "s3"]``).
        new_skill_id:   ID for the resulting macro-skill node.
        name:           Optional human-readable name; defaults to
                        ``"macro_{new_skill_id}"``.

    Returns:
        The newly created macro-skill :class:`SkillNode`.

    Raises:
        ValueError: If ``skill_sequence`` has fewer than 2 elements,
                    or if ``new_skill_id`` already exists in the graph,
                    or if any skill_id in the sequence is missing.
    """
    # ── Validation ───────────────────────────────────────────────────
    if len(skill_sequence) < 2:
        raise ValueError("Skill sequence must contain at least 2 skills.")
    if graph.has_skill(new_skill_id):
        raise ValueError(f"Skill ID '{new_skill_id}' already exists in graph.")

    seq_set: Set[str] = set(skill_sequence)
    for sid in skill_sequence:
        if not graph.has_skill(sid):
            raise ValueError(f"Skill '{sid}' not found in graph.")
    if len(seq_set) != len(skill_sequence):
        raise ValueError("Skill sequence contains duplicates.")

    # ── 0. Collect original skills ───────────────────────────────────
    skills: List[SkillNode] = [graph.get_skill(sid) for sid in skill_sequence]
    first_skill = skills[0]
    last_skill = skills[-1]

    first_id = skill_sequence[0]
    last_id = skill_sequence[-1]

    # ── 1. Build macro-skill node ────────────────────────────────────
    #   policy = composed sequence
    #   initiation_set = first skill's
    #   termination = last skill's
    #   cost = sum of individual costs
    #   reinforcement = average of individual reinforcements
    composed_policy = " → ".join(s.policy for s in skills)

    total_cost = sum(s.cost for s in skills)
    avg_reinforcement = (
        sum(s.reinforcement for s in skills) / len(skills)
    )

    macro_skill = SkillNode(
        skill_id=new_skill_id,
        name=name or f"macro_{new_skill_id}",
        policy=composed_policy,
        termination=last_skill.termination,
        initiation_set=list(first_skill.initiation_set),
        frequency=0,
        reinforcement=avg_reinforcement,
        cost=total_cost,
        version=1,
        utility=0.0,
        tags=["macro", "contracted"] + list(
            set(tag for s in skills for tag in s.tags)
        ),
    )

    # ── 2. Collect edges to rewire ───────────────────────────────────
    #   in-edges:  any edge (X → first_id) where X ∉ seq_set
    #   out-edges: any edge (last_id → Y) where Y ∉ seq_set

    in_edges: List[Dict] = []   # edges to recreate as X → macro
    out_edges: List[Dict] = []  # edges to recreate as macro → Y

    for edge in graph.get_edges():
        src, dst = edge["src"], edge["dst"]

        # In-edge: external → first node of sequence
        if dst == first_id and src not in seq_set:
            in_edges.append(edge)

        # Out-edge: last node of sequence → external
        if src == last_id and dst not in seq_set:
            out_edges.append(edge)

    logger.info(
        "Contracting [%s] → %s  (in-edges: %d, out-edges: %d)",
        " → ".join(skill_sequence), new_skill_id,
        len(in_edges), len(out_edges),
    )

    # ── 3. Remove original nodes (also removes all incident edges) ───
    for sid in skill_sequence:
        graph.remove_skill(sid)

    # ── 4. Add macro-skill node ──────────────────────────────────────
    graph.add_skill(macro_skill)

    # ── 5. Rewire edges ──────────────────────────────────────────────
    for edge in in_edges:
        graph.add_edge(
            src_id=edge["src"],
            dst_id=new_skill_id,
            weight=edge["weight"],
            edge_type=edge["edge_type"],
        )

    for edge in out_edges:
        graph.add_edge(
            src_id=new_skill_id,
            dst_id=edge["dst"],
            weight=edge["weight"],
            edge_type=edge["edge_type"],
        )

    logger.info(
        "Contraction complete: %s (policy=%d chars, cost=%.1f)",
        new_skill_id, len(composed_policy), total_cost,
    )

    return macro_skill
