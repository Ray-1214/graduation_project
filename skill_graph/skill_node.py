"""
SkillNode — formal representation of a reusable skill in the
Self-Evolving Skill Graph.

A skill is a tuple  σ = (π_σ, β_σ, I_σ, μ_σ)  where:

  π_σ : S → A*    policy (natural-language prompt template)
  β_σ : S → {0,1} termination predicate (string condition)
  I_σ ⊆ S         initiation set (applicable task tags / keywords)
  μ_σ = (f, r, c, v)  metadata vector
        f ∈ ℕ      usage frequency
        r ∈ ℝ≥0    cumulative reinforcement
        c ∈ ℝ>0    computational cost (seconds)
        v ∈ ℕ      version counter

Utility:
  U(σ)  = α·r + β·f − γ_c·c            (α, β, γ_c > 0)
  Decay:  U_{t+1} = (1−γ)·U_t + ΔU_t   (γ ∈ (0,1))

Downstream consumers: SkillAbstractor, EvolutionOperator, MetricsTracker.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _generate_skill_id() -> str:
    """Generate a short unique skill identifier."""
    return f"sk-{uuid.uuid4().hex[:10]}"


@dataclass
class SkillNode:
    """A node in the Self-Evolving Skill Graph.

    Represents a reusable, evolvable skill that an LLM agent can
    accumulate through structured memory — without updating model
    weights.

    Attributes:
        skill_id:       Unique identifier for this skill.
        name:           Human-readable skill name.
        policy:         π_σ — natural-language prompt template that
                        maps state to an action sequence.
        termination:    β_σ — string description of the termination
                        predicate (when the skill is "done").
        initiation_set: I_σ — tags / keywords describing the task
                        types this skill is applicable to.
        frequency:      f_σ — how many times this skill has been used.
        reinforcement:  r_σ — cumulative positive reinforcement signal.
        cost:           c_σ — average computational cost in seconds.
        version:        v_σ — version counter (incremented on mutation).
        utility:        Cached utility value U(σ), updated by
                        :meth:`compute_utility` and :meth:`decay`.
        created_at:     Epoch timestamp of creation.
        updated_at:     Epoch timestamp of last modification.
        parent_id:      skill_id of the ancestor skill (if evolved).
        tags:           Free-form metadata tags for search / grouping.
    """

    # ── Identity ─────────────────────────────────────────────────────
    skill_id: str = field(default_factory=_generate_skill_id)
    name: str = ""

    # ── σ = (π, β, I, μ) ────────────────────────────────────────────
    # π_σ : policy (prompt template)
    policy: str = ""

    # β_σ : termination predicate
    termination: str = "Task is complete or max steps reached."

    # I_σ : initiation set (applicable task types)
    initiation_set: List[str] = field(default_factory=list)

    # μ_σ = (f, r, c, v) : metadata vector
    frequency: int = 0                 # f_σ ∈ ℕ
    reinforcement: float = 0.0         # r_σ ∈ ℝ≥0
    cost: float = 1.0                  # c_σ ∈ ℝ>0
    version: int = 1                   # v_σ ∈ ℕ

    # ── Derived / cached ─────────────────────────────────────────────
    utility: float = 0.0               # U(σ), updated explicitly

    # ── Bookkeeping ──────────────────────────────────────────────────
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # ── Utility computation ──────────────────────────────────────────

    def compute_utility(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma_c: float = 0.1,
    ) -> float:
        """Compute and cache the utility value.

        U(σ) = α · r_σ  +  β · f_σ  −  γ_c · c_σ

        Args:
            alpha:   Weight for cumulative reinforcement (r_σ).
            beta:    Weight for usage frequency (f_σ).
            gamma_c: Penalty weight for computational cost (c_σ).

        Returns:
            The computed utility value (also stored in ``self.utility``).
        """
        self.utility = (
            alpha * self.reinforcement
            + beta * self.frequency
            - gamma_c * self.cost
        )
        return self.utility

    def decay(self, gamma: float = 0.05) -> float:
        """Apply multiplicative utility decay between episodes.

        U_{t+1}(σ) = (1 − γ) · U_t(σ)

        For skills that received reinforcement this episode, add ΔU
        *before* calling decay, or call :meth:`reinforce` first.

        Args:
            gamma: Decay rate γ ∈ (0, 1). Higher = faster forgetting.

        Returns:
            The decayed utility value.
        """
        self.utility *= (1.0 - gamma)
        return self.utility

    # ── Reinforcement update ─────────────────────────────────────────

    def reinforce(self, delta_u: float, cost: float = 0.0) -> None:
        """Record a usage event with reinforcement signal.

        Updates frequency, cumulative reinforcement, running-average
        cost, and the cached utility:

            U_{t+1} = (1 − γ) · U_t + ΔU_t

        (Caller applies decay via :meth:`decay` first if needed.)

        Args:
            delta_u: Incremental reinforcement ΔU_t ≥ 0 for this episode.
            cost:    Computational cost of this invocation (seconds).
        """
        self.frequency += 1
        self.reinforcement += max(delta_u, 0.0)
        if cost > 0:
            # Exponential moving average of cost
            self.cost = 0.8 * self.cost + 0.2 * cost
        self.utility += delta_u
        self.updated_at = time.time()

    # ── Initiation-set matching ──────────────────────────────────────

    def matches(self, task_description: str) -> bool:
        """Check if this skill's initiation set covers the given task.

        Returns True if *any* keyword in ``initiation_set`` appears
        in the task description (case-insensitive).
        """
        task_lower = task_description.lower()
        return any(tag.lower() in task_lower for tag in self.initiation_set)

    # ── Versioning ───────────────────────────────────────────────────

    def evolve(
        self,
        new_policy: str,
        new_termination: Optional[str] = None,
        new_tags: Optional[List[str]] = None,
    ) -> "SkillNode":
        """Create a new version of this skill with a mutated policy.

        Returns a *new* SkillNode with incremented version, inheriting
        metadata but resetting frequency and reinforcement.

        Args:
            new_policy:      Updated prompt template (π_σ').
            new_termination: Updated termination predicate (optional).
            new_tags:        Updated tags (optional, inherits if None).

        Returns:
            A child SkillNode linked to this node via ``parent_id``.
        """
        return SkillNode(
            name=self.name,
            policy=new_policy,
            termination=new_termination or self.termination,
            initiation_set=list(self.initiation_set),
            frequency=0,
            reinforcement=0.0,
            cost=self.cost,
            version=self.version + 1,
            utility=0.0,
            parent_id=self.skill_id,
            tags=new_tags if new_tags is not None else list(self.tags),
        )

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillNode":
        """Deserialize from a dict."""
        return cls(**data)

    def save(self, path: str) -> None:
        """Persist to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "SkillNode":
        """Load from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    # ── Display ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SkillNode(id={self.skill_id!r}, name={self.name!r}, "
            f"v={self.version}, f={self.frequency}, "
            f"r={self.reinforcement:.2f}, U={self.utility:.3f})"
        )
