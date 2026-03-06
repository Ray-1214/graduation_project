"""
Tests for TaskCurriculum — graded task set management.

Covers:
  - Loading tiers from JSON files
  - Four experiment orderings (sequential, shuffled, single-tier, repeated)
  - Edge cases (missing files, invalid tier)
  - Query helpers (available_tiers, total_tasks, get_by_tag)
  - Deep-copy isolation (mutations don't leak)
  - Summary output
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from experiments.task_curriculum import TaskCurriculum


# ── Fixture: create tier JSON files in tmp_path ───────────────────

def _make_tasks(tier: int, count: int = 5) -> list[dict]:
    """Generate *count* synthetic tasks for a given tier."""
    return [
        {
            "id": f"t{tier}-{i:03d}",
            "tier": tier,
            "task_description": f"Tier {tier} task {i}",
            "expected_answer": f"answer_{tier}_{i}",
            "complexity": tier,
            "tags": [f"tag_{tier}", "common"],
            "requires_tools": tier >= 3,
        }
        for i in range(1, count + 1)
    ]


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    """Create a tasks directory with 4 tier files."""
    d = tmp_path / "tasks"
    d.mkdir()
    for tier in range(1, 5):
        path = d / f"tier_{tier}.json"
        path.write_text(json.dumps(_make_tasks(tier)), encoding="utf-8")
    return d


@pytest.fixture
def curriculum(tasks_dir: Path) -> TaskCurriculum:
    """Return a TaskCurriculum loaded from the fixture directory."""
    return TaskCurriculum(str(tasks_dir))


# ═══════════════════════════════════════════════════════════════════
#  Loading
# ═══════════════════════════════════════════════════════════════════

class TestLoading:

    def test_loads_all_tiers(self, curriculum):
        assert curriculum.available_tiers == [1, 2, 3, 4]

    def test_total_tasks(self, curriculum):
        assert curriculum.total_tasks == 20  # 4 tiers × 5

    def test_tier_size(self, curriculum):
        for tier in range(1, 5):
            assert curriculum.tier_size(tier) == 5

    def test_load_tier_returns_tasks(self, curriculum):
        tasks = curriculum.load_tier(1)
        assert len(tasks) == 5
        assert tasks[0]["id"] == "t1-001"

    def test_load_tier_invalid_number(self, curriculum):
        with pytest.raises(ValueError):
            curriculum.load_tier(0)
        with pytest.raises(ValueError):
            curriculum.load_tier(5)

    def test_load_tier_missing_file(self, tmp_path):
        d = tmp_path / "empty_tasks"
        d.mkdir()
        tc = TaskCurriculum(str(d))
        with pytest.raises(FileNotFoundError):
            tc.load_tier(1)

    def test_invalid_json_format(self, tmp_path):
        d = tmp_path / "bad_tasks"
        d.mkdir()
        (d / "tier_1.json").write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="Expected JSON array"):
            TaskCurriculum(str(d))


# ═══════════════════════════════════════════════════════════════════
#  Experiment A: Sequential
# ═══════════════════════════════════════════════════════════════════

class TestSequential:

    def test_returns_all_tasks(self, curriculum):
        tasks = curriculum.get_sequential()
        assert len(tasks) == 20

    def test_tier_order(self, curriculum):
        tasks = curriculum.get_sequential()
        tiers = [t["tier"] for t in tasks]
        # First 5 should be tier 1, next 5 tier 2, etc.
        assert tiers == [1]*5 + [2]*5 + [3]*5 + [4]*5

    def test_deep_copy(self, curriculum):
        """Mutations to returned list don't affect internal state."""
        tasks = curriculum.get_sequential()
        tasks[0]["id"] = "MUTATED"
        fresh = curriculum.get_sequential()
        assert fresh[0]["id"] == "t1-001"


# ═══════════════════════════════════════════════════════════════════
#  Experiment B: Shuffled
# ═══════════════════════════════════════════════════════════════════

class TestShuffled:

    def test_returns_all_tasks(self, curriculum):
        tasks = curriculum.get_shuffled()
        assert len(tasks) == 20

    def test_deterministic_seed(self, curriculum):
        a = curriculum.get_shuffled(seed=42)
        b = curriculum.get_shuffled(seed=42)
        assert [t["id"] for t in a] == [t["id"] for t in b]

    def test_different_seed_different_order(self, curriculum):
        a = curriculum.get_shuffled(seed=1)
        b = curriculum.get_shuffled(seed=2)
        # Extremely unlikely to be the same order
        assert [t["id"] for t in a] != [t["id"] for t in b]

    def test_not_sequential_order(self, curriculum):
        """Shuffled should (almost certainly) differ from sequential."""
        seq = curriculum.get_sequential()
        shuf = curriculum.get_shuffled(seed=42)
        assert [t["id"] for t in shuf] != [t["id"] for t in seq]

    def test_same_content(self, curriculum):
        """Same tasks, just different order."""
        seq_ids = sorted(t["id"] for t in curriculum.get_sequential())
        shuf_ids = sorted(t["id"] for t in curriculum.get_shuffled())
        assert seq_ids == shuf_ids


# ═══════════════════════════════════════════════════════════════════
#  Experiment C: Single Tier
# ═══════════════════════════════════════════════════════════════════

class TestSingleTier:

    def test_returns_correct_tier(self, curriculum):
        tasks = curriculum.get_single_tier(3)
        assert len(tasks) == 5
        assert all(t["tier"] == 3 for t in tasks)

    def test_invalid_tier(self, curriculum):
        with pytest.raises(ValueError):
            curriculum.get_single_tier(99)

    def test_deep_copy(self, curriculum):
        tasks = curriculum.get_single_tier(1)
        tasks[0]["id"] = "MUTATED"
        fresh = curriculum.get_single_tier(1)
        assert fresh[0]["id"] == "t1-001"


# ═══════════════════════════════════════════════════════════════════
#  Experiment D: Repeated
# ═══════════════════════════════════════════════════════════════════

class TestRepeated:

    def test_correct_length(self, curriculum):
        tasks = curriculum.get_repeated(n_repeats=3)
        assert len(tasks) == 60  # 20 × 3

    def test_repeats_content(self, curriculum):
        tasks = curriculum.get_repeated(n_repeats=2)
        first_half = [t["id"] for t in tasks[:20]]
        second_half = [t["id"] for t in tasks[20:]]
        assert first_half == second_half

    def test_single_repeat(self, curriculum):
        tasks = curriculum.get_repeated(n_repeats=1)
        assert len(tasks) == 20

    def test_deep_copy_between_repeats(self, curriculum):
        tasks = curriculum.get_repeated(n_repeats=2)
        tasks[0]["id"] = "MUTATED"
        assert tasks[20]["id"] != "MUTATED"  # second copy unaffected


# ═══════════════════════════════════════════════════════════════════
#  Query helpers
# ═══════════════════════════════════════════════════════════════════

class TestQueries:

    def test_get_by_tag(self, curriculum):
        # "common" tag is on every task
        common = curriculum.get_by_tag("common")
        assert len(common) == 20

    def test_get_by_specific_tag(self, curriculum):
        t3_tasks = curriculum.get_by_tag("tag_3")
        assert len(t3_tasks) == 5
        assert all(t["tier"] == 3 for t in t3_tasks)

    def test_get_by_missing_tag(self, curriculum):
        assert curriculum.get_by_tag("nonexistent") == []

    def test_summary(self, curriculum):
        text = curriculum.summary()
        assert "TaskCurriculum" in text
        assert "20 tasks" in text
        assert "Tier 1" in text
        assert "Tier 4" in text

    def test_repr(self, curriculum):
        r = repr(curriculum)
        assert "TaskCurriculum" in r
        assert "T1=5" in r


# ═══════════════════════════════════════════════════════════════════
#  Real tier files
# ═══════════════════════════════════════════════════════════════════

class TestRealTierFiles:
    """Validate the actual tier JSON files shipped with the project."""

    TASKS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "tasks"

    @pytest.fixture
    def real_curriculum(self):
        if not self.TASKS_DIR.exists():
            pytest.skip("Real tasks directory not found")
        return TaskCurriculum(str(self.TASKS_DIR))

    def test_all_tiers_loaded(self, real_curriculum):
        assert real_curriculum.available_tiers == [1, 2, 3, 4]

    def test_minimum_tasks_per_tier(self, real_curriculum):
        """Each tier should have at least 15 tasks."""
        for tier in range(1, 5):
            assert real_curriculum.tier_size(tier) >= 15, (
                f"Tier {tier} has only {real_curriculum.tier_size(tier)} tasks"
            )

    def test_total_at_least_60(self, real_curriculum):
        assert real_curriculum.total_tasks >= 60

    def test_task_schema(self, real_curriculum):
        """Every task should have the required fields."""
        required_keys = {"id", "tier", "task_description", "expected_answer",
                         "complexity", "tags", "requires_tools"}
        for task in real_curriculum.get_sequential():
            missing = required_keys - set(task.keys())
            assert not missing, f"Task {task.get('id', '?')} missing: {missing}"

    def test_unique_ids(self, real_curriculum):
        """All task IDs should be unique."""
        ids = [t["id"] for t in real_curriculum.get_sequential()]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_tier_matches_field(self, real_curriculum):
        """Each task's tier field should match the file it came from."""
        for tier in range(1, 5):
            tasks = real_curriculum.get_single_tier(tier)
            for t in tasks:
                assert t["tier"] == tier, (
                    f"Task {t['id']} in tier {tier} file has tier={t['tier']}"
                )
