"""
TaskCurriculum — graded task sets for systematic skill graph evaluation.

Inspired by Agent0's Curriculum Agent (arXiv:2511.16043) and
Agent0-VL's iterative difficulty scaling (arXiv:2511.19900).

Manages four difficulty tiers:
  Tier 1  Simple    — single-step direct answers
  Tier 2  Moderate  — multi-step reasoning
  Tier 3  Complex   — tool-assisted multi-step
  Tier 4  Compound  — cross-domain integration

Supports four experiment orderings:
  A  Sequential   — tier 1→2→3→4
  B  Shuffled     — random permutation (non-stationarity robustness)
  C  Single-tier  — isolate one tier (pattern exhaustion / saturation)
  D  Repeated     — multiple passes (skill reuse, ρ growth)
"""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Number of tiers defined in the curriculum
NUM_TIERS = 4


class TaskCurriculum:
    """Manage graded task sets for evaluation experiments.

    Args:
        tasks_dir: Path to the directory containing ``tier_{n}.json``
                   files (n = 1 … 4).
    """

    def __init__(self, tasks_dir: str) -> None:
        self.tasks_dir = Path(tasks_dir)
        self._tiers: Dict[int, List[Dict[str, Any]]] = {}
        self._load_all()

    # ── Loading ───────────────────────────────────────────────────────

    def _load_all(self) -> None:
        """Eagerly load all available tier files."""
        for tier in range(1, NUM_TIERS + 1):
            path = self.tasks_dir / f"tier_{tier}.json"
            if path.exists():
                self._tiers[tier] = self._read_json(path)
                logger.info(
                    "Loaded tier %d: %d tasks from %s",
                    tier, len(self._tiers[tier]), path,
                )

    @staticmethod
    def _read_json(path: Path) -> List[Dict[str, Any]]:
        """Read and validate a tier JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
        return data

    def load_tier(self, tier: int) -> List[Dict[str, Any]]:
        """Load (or re-load) a specific tier.

        Args:
            tier: Tier number (1–4).

        Returns:
            List of task dicts for the tier.

        Raises:
            FileNotFoundError: If the tier file does not exist.
            ValueError: If tier is out of range.
        """
        if tier < 1 or tier > NUM_TIERS:
            raise ValueError(
                f"Tier must be 1–{NUM_TIERS}, got {tier}"
            )
        path = self.tasks_dir / f"tier_{tier}.json"
        if not path.exists():
            raise FileNotFoundError(f"Tier file not found: {path}")
        self._tiers[tier] = self._read_json(path)
        return copy.deepcopy(self._tiers[tier])

    # ── Experiment orderings ──────────────────────────────────────────

    def get_sequential(self) -> List[Dict[str, Any]]:
        """Experiment A: all tasks in tier order (1→2→3→4).

        Verifies the expected progression: base skills → edges →
        contraction → macro-skill formation.
        """
        result: List[Dict[str, Any]] = []
        for tier in sorted(self._tiers.keys()):
            result.extend(copy.deepcopy(self._tiers[tier]))
        return result

    def get_shuffled(self, seed: int = 42) -> List[Dict[str, Any]]:
        """Experiment B: all tasks in random order.

        Verifies non-stationarity robustness — the skill graph
        should still converge even when task difficulty is not
        monotonically increasing.

        Args:
            seed: Random seed for reproducibility.
        """
        all_tasks = self.get_sequential()
        rng = random.Random(seed)
        rng.shuffle(all_tasks)
        return all_tasks

    def get_single_tier(self, tier: int) -> List[Dict[str, Any]]:
        """Experiment C: only tasks from a single tier.

        Verifies pattern exhaustion / quick saturation — repeated
        exposure to the same difficulty level should cause κ→0
        and ΔΣ→0 faster.

        Args:
            tier: Tier number (1–4).

        Raises:
            ValueError: If tier is out of range or not loaded.
        """
        if tier not in self._tiers:
            raise ValueError(
                f"Tier {tier} not loaded. Available: {sorted(self._tiers.keys())}"
            )
        return copy.deepcopy(self._tiers[tier])

    def get_repeated(self, n_repeats: int = 3) -> List[Dict[str, Any]]:
        """Experiment D: full curriculum repeated *n_repeats* times.

        Verifies skill reuse — ρ should increase across repeats
        as macro-skills are reused for familiar patterns.

        Args:
            n_repeats: Number of full passes through the curriculum.
        """
        one_pass = self.get_sequential()
        return [copy.deepcopy(t) for _ in range(n_repeats) for t in one_pass]

    # ── Queries ───────────────────────────────────────────────────────

    @property
    def available_tiers(self) -> List[int]:
        """List of tier numbers with loaded tasks."""
        return sorted(self._tiers.keys())

    @property
    def total_tasks(self) -> int:
        """Total number of tasks across all loaded tiers."""
        return sum(len(tasks) for tasks in self._tiers.values())

    def tier_size(self, tier: int) -> int:
        """Number of tasks in a specific tier."""
        return len(self._tiers.get(tier, []))

    def get_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Return all tasks matching a tag, across all tiers."""
        result: List[Dict[str, Any]] = []
        for tier in sorted(self._tiers.keys()):
            for task in self._tiers[tier]:
                if tag in task.get("tags", []):
                    result.append(copy.deepcopy(task))
        return result

    def summary(self) -> str:
        """Return a text summary of the loaded curriculum."""
        lines = [f"TaskCurriculum ({self.total_tasks} tasks)"]
        for tier in sorted(self._tiers.keys()):
            tasks = self._tiers[tier]
            tags = set()
            for t in tasks:
                tags.update(t.get("tags", []))
            tool_count = sum(
                1 for t in tasks if t.get("requires_tools", False)
            )
            lines.append(
                f"  Tier {tier}: {len(tasks)} tasks "
                f"({tool_count} require tools)  "
                f"tags={sorted(tags)}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        tier_info = ", ".join(
            f"T{t}={len(self._tiers[t])}"
            for t in sorted(self._tiers.keys())
        )
        return f"TaskCurriculum({tier_info})"
