"""
MemoryPartition — three-tier structured skill memory management.

Tiers (ordered by utility):
  M_active   — working-memory candidates, highest priority retrieval
  M_cold     — lower-priority but still retrievable
  M_archive  — excluded from default retrieval, never deleted

Tier transitions with hysteresis (prevents oscillation at boundaries):

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   archive ──U > θ_low+ε_l──▸ cold ──U ≥ θ_high+ε_h──▸ active  │
  │   archive ◂──U ≤ θ_low−ε_l── cold ◂──U < θ_high−ε_h── active  │
  │                                                                 │
  │   Hysteresis margins ensure that a skill must cross a wider     │
  │   band before changing tier, preventing rapid flip-flop.        │
  └─────────────────────────────────────────────────────────────────┘

  Promotion thresholds (entering from below):
    cold → active  :  U ≥ θ_high + ε_h
    archive → cold :  U > θ_low  + ε_l

  Demotion thresholds (dropping from above):
    active → cold  :  U < θ_high − ε_h
    cold → archive :  U ≤ θ_low  − ε_l

  Direct jumps:
    archive → active : U ≥ θ_high + ε_h  (skip cold)
    active → archive : U ≤ θ_low  − ε_l  (skip cold)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal

if TYPE_CHECKING:
    from skill_graph.skill_graph import SkillGraph

from skill_graph.skill_node import SkillNode

Tier = Literal["active", "cold", "archive"]

# Default tier for skills not yet assigned
_DEFAULT_TIER: Tier = "cold"


@dataclass
class MemoryPartition:
    """Manages the three-tier memory partition for skill nodes.

    Args:
        theta_high: Upper utility threshold for active promotion.
        theta_low:  Lower utility threshold for archive demotion.
        epsilon_h:  Hysteresis margin for the active boundary.
        epsilon_l:  Hysteresis margin for the archive boundary.
    """

    theta_high: float = 0.7
    theta_low: float = 0.3
    epsilon_h: float = 0.1
    epsilon_l: float = 0.1

    # Internal mapping: skill_id → current tier
    _tiers: Dict[str, Tier] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.theta_high <= self.theta_low:
            raise ValueError(
                f"theta_high ({self.theta_high}) must be > "
                f"theta_low ({self.theta_low})"
            )

    # ── Core tier assignment ─────────────────────────────────────────

    def assign_tier(self, skill: SkillNode, current_tier: Tier) -> Tier:
        """Determine the new tier for a skill based on its utility and
        current tier, applying hysteresis rules.

        Promotion thresholds (must *exceed* to move up):
          • cold/archive → active : U ≥ θ_high + ε_h
          • archive → cold        : U > θ_low  + ε_l

        Demotion thresholds (must *fall below* to move down):
          • active → cold         : U < θ_high − ε_h
          • cold → archive        : U ≤ θ_low  − ε_l

        Args:
            skill: The SkillNode to evaluate.
            current_tier: The tier the skill is currently in.

        Returns:
            The (possibly unchanged) tier.
        """
        u = skill.utility

        promote_to_active = self.theta_high + self.epsilon_h
        demote_from_active = self.theta_high - self.epsilon_h
        promote_from_archive = self.theta_low + self.epsilon_l
        demote_to_archive = self.theta_low - self.epsilon_l

        if current_tier == "active":
            # Can only leave active if utility drops well below θ_high
            if u <= demote_to_archive:
                return "archive"
            if u < demote_from_active:
                return "cold"
            return "active"

        elif current_tier == "cold":
            # Promote to active?
            if u >= promote_to_active:
                return "active"
            # Demote to archive?
            if u <= demote_to_archive:
                return "archive"
            return "cold"

        else:  # archive
            # Promote to active? (direct jump)
            if u >= promote_to_active:
                return "active"
            # Promote to cold?
            if u > promote_from_archive:
                return "cold"
            return "archive"

    # ── Bulk update ──────────────────────────────────────────────────

    def update_all(self, graph: "SkillGraph") -> Dict[str, Tier]:
        """Re-evaluate tiers for every skill in the graph.

        Args:
            graph: The SkillGraph whose skills to partition.

        Returns:
            Mapping ``{skill_id: new_tier}`` for every skill.
        """
        result: Dict[str, Tier] = {}
        for skill in graph.skills:
            old_tier = self._tiers.get(skill.skill_id, _DEFAULT_TIER)
            new_tier = self.assign_tier(skill, old_tier)
            self._tiers[skill.skill_id] = new_tier
            result[skill.skill_id] = new_tier
        return result

    # ── Queries ──────────────────────────────────────────────────────

    def get_tier(self, skill_id: str) -> Tier:
        """Return the current tier of a skill.

        Returns ``"cold"`` for untracked skills.
        """
        return self._tiers.get(skill_id, _DEFAULT_TIER)

    def get_skills_by_tier(self, tier: Tier) -> List[str]:
        """Return skill IDs belonging to the given tier."""
        return [sid for sid, t in self._tiers.items() if t == tier]

    # ── Manual tier management ───────────────────────────────────────

    def set_tier(self, skill_id: str, tier: Tier) -> None:
        """Manually override a skill's tier."""
        self._tiers[skill_id] = tier

    def remove(self, skill_id: str) -> None:
        """Stop tracking a skill (e.g. after graph removal)."""
        self._tiers.pop(skill_id, None)

    # ── Diagnostics ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, int]:
        """Return counts per tier."""
        counts = {"active": 0, "cold": 0, "archive": 0}
        for t in self._tiers.values():
            counts[t] += 1
        return counts

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"MemoryPartition(active={s['active']}, "
            f"cold={s['cold']}, archive={s['archive']})"
        )
