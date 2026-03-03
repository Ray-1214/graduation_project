"""
SkillRetriever — activation-based skill retrieval from the graph.

Given a structured task description T and a SkillGraph G_t, the
retriever computes an activation score for each skill:

    A(σ) = λ₁·sim(T, σ) + λ₂·U(σ) + λ₃·centrality(σ, G_t)

where:
  sim         — textual similarity between task and skill
  U(σ)        — normalised utility ∈ [0, 1]
  centrality  — normalised in-degree centrality in G_t
  λ₁+λ₂+λ₃   = 1

The top-k skills are returned with their activation scores.
When ``skill.document_path`` points to an existing ``.md`` file,
the full document is loaded and injected into the prompt instead
of the short ``skill.policy`` string.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx  # type: ignore

from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RetrievedSkill:
    """A single skill returned by the retriever.

    Attributes:
        skill_id:         Unique ID of the skill.
        name:             Human-readable skill name.
        policy:           Short policy string.
        activation_score: Computed A(σ).
        document:         Full ``.md`` content (or fallback to policy).
    """
    skill_id: str
    name: str
    policy: str
    activation_score: float
    document: str


# ═══════════════════════════════════════════════════════════════════
#  SkillRetriever
# ═══════════════════════════════════════════════════════════════════

class SkillRetriever:
    """Retrieves the most relevant skills from a SkillGraph.

    Args:
        top_k:   Number of skills to retrieve.
        lambda1: Weight for textual similarity.
        lambda2: Weight for normalised utility.
        lambda3: Weight for normalised in-degree centrality.
        root:    Project root directory (for resolving document_path).
    """

    def __init__(
        self,
        top_k: int = 3,
        lambda1: float = 0.5,
        lambda2: float = 0.3,
        lambda3: float = 0.2,
        root: Optional[Path] = None,
    ) -> None:
        if abs((lambda1 + lambda2 + lambda3) - 1.0) > 1e-6:
            raise ValueError(
                f"λ weights must sum to 1.0, got "
                f"{lambda1}+{lambda2}+{lambda3}={lambda1+lambda2+lambda3}"
            )
        self.top_k = top_k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.root = root or Path.cwd()

    # ── Public API ───────────────────────────────────────────────────

    def retrieve(
        self,
        task: str,
        graph: SkillGraph,
    ) -> List[RetrievedSkill]:
        """Compute activation scores and return top-k skills.

        Args:
            task:  Task description string.
            graph: The skill graph to search.

        Returns:
            List of :class:`RetrievedSkill`, sorted descending by
            activation score.  Empty list if graph is empty.
        """
        if len(graph) == 0:
            return []

        # Pre-compute centrality (normalised in-degree)
        centrality = self._compute_centrality(graph)

        # Normalise utilities to [0, 1]
        utilities = [s.utility for s in graph.skills]
        max_u = max(utilities) if utilities else 1.0
        max_u = max(max_u, 1e-9)

        task_lower = task.lower()
        scored: List[Tuple[SkillNode, float]] = []

        for skill in graph.skills:
            sim = self._similarity(task_lower, skill)
            norm_u = skill.utility / max_u
            cent = centrality.get(skill.skill_id, 0.0)

            activation = (
                self.lambda1 * sim
                + self.lambda2 * norm_u
                + self.lambda3 * cent
            )
            scored.append((skill, activation))

        # Sort descending, take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.top_k]

        # Build result with document loading
        results: List[RetrievedSkill] = []
        for skill, act_score in top:
            doc = self._load_document(skill)
            results.append(RetrievedSkill(
                skill_id=skill.skill_id,
                name=skill.name,
                policy=skill.policy,
                activation_score=act_score,
                document=doc,
            ))

        return results

    def format_for_prompt(
        self,
        retrieved: List[RetrievedSkill],
    ) -> str:
        """Format retrieved skills as a prompt injection block.

        When a ``.md`` document is available, the full content is used.
        Otherwise, falls back to the short policy string.
        """
        if not retrieved:
            return ""

        blocks: List[str] = []
        for r in retrieved:
            header = (
                f"=== 已知技能：{r.name}"
                f"（相關度：{r.activation_score:.2f}）==="
            )
            footer = "=" * len(header)
            blocks.append(f"{header}\n{r.document}\n{footer}")

        return "\n\n".join(blocks)

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _compute_centrality(graph: SkillGraph) -> Dict[str, float]:
        """Compute normalised in-degree centrality for all nodes."""
        try:
            return nx.in_degree_centrality(graph._graph)
        except nx.NetworkXError:
            return {nid: 0.0 for nid in graph._graph.nodes}

    @staticmethod
    def _similarity(task_lower: str, skill: SkillNode) -> float:
        """Textual similarity between task and skill.

        Combines initiation-set keyword overlap with policy similarity
        via SequenceMatcher.
        """
        # Build a combined skill text
        skill_text = " ".join(
            skill.initiation_set + [skill.name, skill.policy]
        ).lower()
        return SequenceMatcher(None, task_lower, skill_text).ratio()

    def _load_document(self, skill: SkillNode) -> str:
        """Load the full .md document for a skill, with fallback."""
        if skill.document_path:
            doc_path = self.root / skill.document_path
            try:
                if doc_path.exists():
                    content = doc_path.read_text(encoding="utf-8").strip()
                    if content:
                        return content
                    logger.warning(
                        "Skill doc '%s' is empty, using policy fallback.",
                        doc_path,
                    )
                else:
                    logger.warning(
                        "Skill doc '%s' not found, using policy fallback.",
                        doc_path,
                    )
            except OSError as exc:
                logger.warning(
                    "Failed to read skill doc '%s': %s", doc_path, exc,
                )

        # Fallback: structured policy string
        lines = [
            f"技能: {skill.name}",
            f"策略: {skill.policy}",
            f"適用: {', '.join(skill.initiation_set)}",
            f"終止: {skill.termination}",
        ]
        return "\n".join(lines)

    # ── Display ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SkillRetriever(top_k={self.top_k}, "
            f"λ=({self.lambda1},{self.lambda2},{self.lambda3}))"
        )
