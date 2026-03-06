"""
Tests for MetricsTracker — five theoretical evaluation criteria.

Covers:
  - Single record computation correctness
  - Edge cases (empty traces, zero raw lengths, single episode)
  - κ sliding window behaviour
  - ΔΣ first-episode and multi-episode
  - Iteration summary grouping and aggregation
  - CSV / JSON export format
  - summary() trend detection and theory compliance
"""

from __future__ import annotations

import csv
import json
import math
from unittest.mock import MagicMock

import pytest

from skill_graph.metrics import MetricsTracker, _linear_slope, _trend_symbol
from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_graph(n_skills: int = 3, utilities: list[float] | None = None) -> SkillGraph:
    """Build a small SkillGraph with *n_skills* nodes."""
    g = SkillGraph(capacity=100)
    if utilities is None:
        utilities = [1.0 / n_skills] * n_skills
    for i, u in enumerate(utilities):
        node = SkillNode(
            name=f"skill_{i}",
            policy=f"Do thing {i}",
            termination=f"done_{i}",
            initiation_set=[f"task_{i}"],
        )
        node.utility = u
        g.add_skill(node)
    return g


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_linear_slope_empty(self):
        assert _linear_slope([]) == 0.0

    def test_linear_slope_single(self):
        assert _linear_slope([5.0]) == 0.0

    def test_linear_slope_increasing(self):
        slope = _linear_slope([1.0, 2.0, 3.0, 4.0, 5.0])
        assert slope == pytest.approx(1.0)

    def test_linear_slope_decreasing(self):
        slope = _linear_slope([5.0, 4.0, 3.0, 2.0, 1.0])
        assert slope == pytest.approx(-1.0)

    def test_linear_slope_flat(self):
        slope = _linear_slope([3.0, 3.0, 3.0])
        assert slope == pytest.approx(0.0)

    def test_trend_symbol_up(self):
        assert _trend_symbol(0.01) == "↑"

    def test_trend_symbol_down(self):
        assert _trend_symbol(-0.01) == "↓"

    def test_trend_symbol_stable(self):
        assert _trend_symbol(0.00001) == "→"


# ═══════════════════════════════════════════════════════════════════
#  Core record()
# ═══════════════════════════════════════════════════════════════════

class TestRecord:

    def test_rho_basic(self):
        """ρ = 1 - compressed/raw."""
        tracker = MetricsTracker()
        g = _make_graph(3, [0.5, 0.3, 0.2])
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[10, 20],
            compressed_trace_lengths=[6, 12],
        )
        # ρ = 1 - 18/30 = 0.4
        assert rec["rho"] == pytest.approx(0.4)

    def test_rho_zero_raw(self):
        """raw_sum = 0 → ρ = 0."""
        tracker = MetricsTracker()
        g = _make_graph(1)
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[0, 0],
            compressed_trace_lengths=[0, 0],
        )
        assert rec["rho"] == 0.0

    def test_rho_empty_traces(self):
        """Empty trace lists → ρ = 0."""
        tracker = MetricsTracker()
        g = _make_graph(1)
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[],
            compressed_trace_lengths=[],
        )
        assert rec["rho"] == 0.0

    def test_kappa_single_episode(self):
        """κ = contraction_ops / total_graph_ops for a single episode."""
        tracker = MetricsTracker()
        g = _make_graph(2)
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[10],
            compressed_trace_lengths=[8],
            contraction_ops=3,
            total_graph_ops=10,
        )
        assert rec["kappa"] == pytest.approx(0.3)

    def test_kappa_zero_ops(self):
        """No graph ops → κ = 0."""
        tracker = MetricsTracker()
        g = _make_graph(1)
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[10],
            compressed_trace_lengths=[8],
            contraction_ops=0, total_graph_ops=0,
        )
        assert rec["kappa"] == 0.0

    def test_entropy_from_graph(self):
        """H comes from graph.compute_entropy()."""
        g = _make_graph(3, [0.5, 0.3, 0.2])
        expected_H = g.compute_entropy()
        tracker = MetricsTracker()
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[10],
            compressed_trace_lengths=[8],
        )
        assert rec["entropy"] == pytest.approx(expected_H)

    def test_delta_sigma_first_episode(self):
        """First episode ΔΣ = |Σ|."""
        tracker = MetricsTracker()
        g = _make_graph(5)
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[10],
            compressed_trace_lengths=[8],
        )
        assert rec["delta_sigma"] == 5
        assert rec["sigma_size"] == 5

    def test_delta_sigma_multi_episode(self):
        """ΔΣ = current |Σ| - previous |Σ|."""
        tracker = MetricsTracker()
        g1 = _make_graph(3)
        tracker.record(0, g1, [10], [8])

        g2 = _make_graph(5)
        rec2 = tracker.record(1, g2, [10], [8])
        assert rec2["delta_sigma"] == 2  # 5 - 3

        g3 = _make_graph(4)
        rec3 = tracker.record(2, g3, [10], [8])
        assert rec3["delta_sigma"] == -1  # 4 - 5

    def test_planning_depth(self):
        """E[D] = mean(compressed_trace_lengths)."""
        tracker = MetricsTracker()
        g = _make_graph(2)
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[10, 20],
            compressed_trace_lengths=[6, 12],
        )
        assert rec["planning_depth"] == pytest.approx(9.0)  # (6+12)/2

    def test_planning_depth_empty(self):
        """Empty compressed traces → E[D] = 0."""
        tracker = MetricsTracker()
        g = _make_graph(1)
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[],
            compressed_trace_lengths=[],
        )
        assert rec["planning_depth"] == 0.0

    def test_timestamp_present(self):
        """Record contains an ISO timestamp."""
        tracker = MetricsTracker()
        g = _make_graph(1)
        rec = tracker.record(0, g, [10], [8])
        assert "timestamp" in rec
        assert "T" in rec["timestamp"]  # ISO format


# ═══════════════════════════════════════════════════════════════════
#  κ Sliding Window
# ═══════════════════════════════════════════════════════════════════

class TestKappaWindow:

    def test_window_accumulates(self):
        """κ uses ops from the last `contraction_window` episodes."""
        tracker = MetricsTracker(contraction_window=3)
        g = _make_graph(2)

        # Episode 0: 2/10
        tracker.record(0, g, [10], [8], contraction_ops=2, total_graph_ops=10)
        # Episode 1: 3/10
        tracker.record(1, g, [10], [8], contraction_ops=3, total_graph_ops=10)
        # Episode 2: 1/5 — window covers eps 0,1,2
        rec = tracker.record(2, g, [10], [8], contraction_ops=1, total_graph_ops=5)
        # κ = (2+3+1) / (10+10+5) = 6/25
        assert rec["kappa"] == pytest.approx(6.0 / 25.0)

    def test_window_slides(self):
        """Old episodes fall out of the window."""
        tracker = MetricsTracker(contraction_window=2)
        g = _make_graph(2)

        # Episode 0: 5/10 (will fall out)
        tracker.record(0, g, [10], [8], contraction_ops=5, total_graph_ops=10)
        # Episode 1: 1/10
        tracker.record(1, g, [10], [8], contraction_ops=1, total_graph_ops=10)
        # Episode 2: 2/10 — window covers eps 1,2
        rec = tracker.record(2, g, [10], [8], contraction_ops=2, total_graph_ops=10)
        # κ = (1+2) / (10+10) = 3/20
        assert rec["kappa"] == pytest.approx(3.0 / 20.0)


# ═══════════════════════════════════════════════════════════════════
#  get_history
# ═══════════════════════════════════════════════════════════════════

class TestGetHistory:

    def test_returns_all_records(self):
        tracker = MetricsTracker()
        g = _make_graph(2)
        tracker.record(0, g, [10], [8])
        tracker.record(1, g, [10], [6])
        tracker.record(2, g, [10], [4])

        history = tracker.get_history()
        assert len(history) == 3
        assert [r["episode_id"] for r in history] == [0, 1, 2]

    def test_returns_copy(self):
        """Modifying returned list doesn't affect internal state."""
        tracker = MetricsTracker()
        g = _make_graph(1)
        tracker.record(0, g, [10], [8])
        history = tracker.get_history()
        history.clear()
        assert len(tracker.get_history()) == 1


# ═══════════════════════════════════════════════════════════════════
#  Iteration Summary
# ═══════════════════════════════════════════════════════════════════

class TestIterationSummary:

    def test_empty(self):
        tracker = MetricsTracker()
        assert tracker.get_iteration_summary() == []

    def test_grouping(self):
        """20 episodes with episodes_per_iteration=10 → 2 iterations."""
        tracker = MetricsTracker()
        g = _make_graph(3, [0.5, 0.3, 0.2])
        for i in range(20):
            tracker.record(i, g, [10], [8 - i % 3])

        summaries = tracker.get_iteration_summary(episodes_per_iteration=10)
        assert len(summaries) == 2

        assert summaries[0]["iteration"] == 0
        assert summaries[0]["episodes"] == (0, 9)
        assert summaries[1]["iteration"] == 1
        assert summaries[1]["episodes"] == (10, 19)

    def test_partial_last_iteration(self):
        """7 episodes with episodes_per_iteration=5 → 2 iterations."""
        tracker = MetricsTracker()
        g = _make_graph(2)
        for i in range(7):
            tracker.record(i, g, [10], [8])

        summaries = tracker.get_iteration_summary(episodes_per_iteration=5)
        assert len(summaries) == 2
        assert summaries[1]["episodes"] == (5, 6)

    def test_aggregation_keys(self):
        """Each iteration summary has the expected keys."""
        tracker = MetricsTracker()
        g = _make_graph(2)
        for i in range(5):
            tracker.record(i, g, [10], [8])

        summaries = tracker.get_iteration_summary(episodes_per_iteration=5)
        assert len(summaries) == 1
        s = summaries[0]
        assert "iteration" in s
        assert "episodes" in s
        assert "mean_rho" in s
        assert "mean_kappa" in s
        assert "last_entropy" in s
        assert "last_sigma_size" in s
        assert "mean_depth" in s


# ═══════════════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════════════

class TestExport:

    def test_csv_format(self, tmp_path):
        """CSV has correct headers and rows."""
        tracker = MetricsTracker()
        g = _make_graph(2)
        tracker.record(0, g, [10], [8])
        tracker.record(1, g, [10], [6])

        path = str(tmp_path / "metrics.csv")
        tracker.export_csv(path)

        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert set(rows[0].keys()) == {
            "episode_id", "timestamp", "rho", "kappa",
            "entropy", "delta_sigma", "planning_depth", "sigma_size",
        }
        assert rows[0]["episode_id"] == "0"
        assert rows[1]["episode_id"] == "1"

    def test_json_format(self, tmp_path):
        """JSON is a clean array without internal fields."""
        tracker = MetricsTracker()
        g = _make_graph(2)
        tracker.record(0, g, [10], [8])

        path = str(tmp_path / "metrics.json")
        tracker.export_json(path)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert isinstance(data, list)
        assert len(data) == 1
        rec = data[0]
        # No internal fields
        assert not any(k.startswith("_") for k in rec.keys())
        assert "rho" in rec
        assert "episode_id" in rec

    def test_json_no_internal_fields(self, tmp_path):
        """Exported JSON must not contain _contraction_ops etc."""
        tracker = MetricsTracker()
        g = _make_graph(2)
        tracker.record(0, g, [10], [8], contraction_ops=3, total_graph_ops=10)

        path = str(tmp_path / "metrics.json")
        tracker.export_json(path)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert "_contraction_ops" not in data[0]
        assert "_total_graph_ops" not in data[0]


# ═══════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════

class TestSummary:

    def test_empty_summary(self):
        tracker = MetricsTracker()
        assert "No metrics" in tracker.summary()

    def test_summary_contains_metrics(self):
        """Summary mentions all five metrics."""
        tracker = MetricsTracker()
        g = _make_graph(3, [0.5, 0.3, 0.2])
        for i in range(5):
            tracker.record(i, g, [10 + i], [8])

        text = tracker.summary()
        assert "ρ" in text
        assert "κ" in text
        assert "H" in text
        assert "ΔΣ" in text
        assert "E[D]" in text

    def test_summary_theory_compliance(self):
        """Summary includes theory compliance section."""
        tracker = MetricsTracker()
        g = _make_graph(3, [0.5, 0.3, 0.2])
        for i in range(5):
            tracker.record(i, g, [10], [8])

        text = tracker.summary()
        assert "Theory Compliance" in text
        # Should have either ✓ or ✗ markers
        assert "✓" in text or "✗" in text

    def test_summary_iteration_table(self):
        """Summary includes iteration summary table when enough data."""
        tracker = MetricsTracker()
        g = _make_graph(3, [0.5, 0.3, 0.2])
        for i in range(15):
            tracker.record(i, g, [10], [8])

        text = tracker.summary()
        assert "Iteration Summary" in text

    def test_decreasing_depth_detected(self):
        """If E[D] decreases over time, slope should be negative."""
        tracker = MetricsTracker()
        g = _make_graph(3, [0.5, 0.3, 0.2])
        for i in range(10):
            # Decreasing compressed trace length
            tracker.record(i, g, [20], [20 - i])

        text = tracker.summary()
        # The depth line should show ↓
        assert "↓" in text


# ═══════════════════════════════════════════════════════════════════
#  repr
# ═══════════════════════════════════════════════════════════════════

class TestRepr:

    def test_repr(self):
        tracker = MetricsTracker()
        assert "MetricsTracker" in repr(tracker)
        assert "episodes=0" in repr(tracker)
