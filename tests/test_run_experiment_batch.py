"""
Tests for run_experiment_batch.py

Since the full pipeline requires an LLM, these tests focus on:
  - CLI argument parsing
  - Helper functions (_simple_accuracy, _format_duration)
  - Output generation (_build_summary, _write_iteration_csv)
  - Task loading and mode selection
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.run_experiment_batch import (
    _simple_accuracy,
    _format_duration,
    _build_summary,
    _write_iteration_csv,
    build_parser,
)
from skill_graph.metrics import MetricsTracker
from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode


# ═══════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════

class TestSimpleAccuracy:

    def test_exact_match(self):
        assert _simple_accuracy("345", "345") is True

    def test_contained_in_answer(self):
        assert _simple_accuracy("The answer is 345.", "345") is True

    def test_case_insensitive(self):
        assert _simple_accuracy("paris", "Paris") is True

    def test_no_match(self):
        assert _simple_accuracy("London", "Paris") is False

    def test_empty_expected(self):
        assert _simple_accuracy("anything", "") is False

    def test_empty_answer(self):
        assert _simple_accuracy("", "345") is False


class TestFormatDuration:

    def test_less_than_minute(self):
        assert _format_duration(45.3) == "0m 45s"

    def test_minutes_and_seconds(self):
        assert _format_duration(125.7) == "2m 5s"

    def test_zero(self):
        assert _format_duration(0) == "0m 0s"

    def test_exact_minutes(self):
        assert _format_duration(120.0) == "2m 0s"


# ═══════════════════════════════════════════════════════════════════
#  CLI parser
# ═══════════════════════════════════════════════════════════════════

class TestParser:

    def test_required_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-dir", "tasks/",
            "--experiment-mode", "sequential",
            "--output-dir", "output/",
        ])
        assert args.tasks_dir == "tasks/"
        assert args.experiment_mode == "sequential"
        assert args.output_dir == "output/"

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-dir", "t",
            "--experiment-mode", "shuffled",
            "--output-dir", "o",
        ])
        assert args.seed == 42
        assert args.snapshot_interval == 10
        assert args.episodes_per_iteration == 10
        assert args.gamma == 0.05
        assert args.theta_high == 0.7
        assert args.theta_low == 0.3
        assert args.delta == 3
        assert args.repeats == 3
        assert args.tier is None
        assert args.verbose is False

    def test_all_modes_accepted(self):
        parser = build_parser()
        for mode in ["sequential", "shuffled", "single_tier", "repeated"]:
            args = parser.parse_args([
                "--tasks-dir", "t",
                "--experiment-mode", mode,
                "--output-dir", "o",
            ])
            assert args.experiment_mode == mode

    def test_invalid_mode_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--tasks-dir", "t",
                "--experiment-mode", "invalid",
                "--output-dir", "o",
            ])

    def test_custom_params(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-dir", "t",
            "--experiment-mode", "single_tier",
            "--output-dir", "o",
            "--tier", "3",
            "--seed", "123",
            "--gamma", "0.1",
            "--theta-high", "0.8",
            "--theta-low", "0.2",
            "--delta", "5",
            "--contraction-window", "20",
            "--snapshot-interval", "5",
            "--episodes-per-iteration", "15",
            "--repeats", "5",
            "-v",
        ])
        assert args.tier == 3
        assert args.seed == 123
        assert args.gamma == 0.1
        assert args.theta_high == 0.8
        assert args.theta_low == 0.2
        assert args.delta == 5
        assert args.contraction_window == 20
        assert args.snapshot_interval == 5
        assert args.episodes_per_iteration == 15
        assert args.repeats == 5
        assert args.verbose is True


# ═══════════════════════════════════════════════════════════════════
#  Iteration CSV
# ═══════════════════════════════════════════════════════════════════

class TestWriteIterationCsv:

    def test_writes_correct_format(self, tmp_path):
        iterations = [
            {
                "iteration": 0,
                "episodes": (0, 9),
                "mean_rho": 0.15,
                "mean_kappa": 0.42,
                "last_entropy": 1.5,
                "last_sigma_size": 5,
                "mean_depth": 11.2,
            },
            {
                "iteration": 1,
                "episodes": (10, 19),
                "mean_rho": 0.28,
                "mean_kappa": 0.14,
                "last_entropy": 2.85,
                "last_sigma_size": 11,
                "mean_depth": 8.4,
            },
        ]

        path = str(tmp_path / "iter.csv")
        _write_iteration_csv(iterations, path)

        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["iteration"] == "0"
        assert rows[0]["ep_start"] == "0"
        assert rows[0]["ep_end"] == "9"
        assert rows[1]["mean_rho"] == "0.2800"

    def test_empty_iterations(self, tmp_path):
        path = str(tmp_path / "empty.csv")
        _write_iteration_csv([], path)

        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 0


# ═══════════════════════════════════════════════════════════════════
#  Summary builder
# ═══════════════════════════════════════════════════════════════════

def _make_graph(n: int = 3) -> SkillGraph:
    g = SkillGraph(capacity=100)
    for i in range(n):
        node = SkillNode(
            name=f"s{i}", policy=f"do {i}",
            termination=f"done {i}", initiation_set=[f"t{i}"],
        )
        node.utility = 1.0 / max(n, 1)
        g.add_skill(node)
    return g


class TestBuildSummary:

    def test_contains_header(self):
        tracker = MetricsTracker()
        g = _make_graph(3)
        for i in range(5):
            tracker.record(i, g, [10], [8])

        text = _build_summary(
            tracker=tracker,
            experiment_mode="sequential",
            total_episodes=5,
            total_duration=120.0,
            seed=42,
            accuracy_by_tier={1: [True, True, False]},
            iterations=tracker.get_iteration_summary(5),
        )
        assert "Self-Evolving Skill Graph" in text
        assert "sequential" in text
        assert "Total Episodes: 5" in text

    def test_contains_accuracy(self):
        tracker = MetricsTracker()
        g = _make_graph(2)
        for i in range(3):
            tracker.record(i, g, [10], [8])

        text = _build_summary(
            tracker=tracker,
            experiment_mode="shuffled",
            total_episodes=3,
            total_duration=60.0,
            seed=42,
            accuracy_by_tier={1: [True, False], 2: [True]},
            iterations=[],
        )
        assert "NOT a formal metric" in text
        assert "Tier 1:" in text
        assert "Tier 2:" in text

    def test_theoretical_verification_present(self):
        tracker = MetricsTracker()
        g = _make_graph(3)
        for i in range(10):
            tracker.record(i, g, [10], [8])

        text = _build_summary(
            tracker=tracker,
            experiment_mode="sequential",
            total_episodes=10,
            total_duration=300.0,
            seed=42,
            accuracy_by_tier={},
            iterations=tracker.get_iteration_summary(5),
        )
        assert "Theoretical Prediction" in text
        # Should mention all 5 metrics
        assert "ρ" in text
        assert "κ" in text
        assert "ΔΣ" in text
        assert "E[D]" in text

    def test_insufficient_data_message(self):
        tracker = MetricsTracker()
        g = _make_graph(1)
        tracker.record(0, g, [10], [8])

        text = _build_summary(
            tracker=tracker,
            experiment_mode="sequential",
            total_episodes=1,
            total_duration=5.0,
            seed=42,
            accuracy_by_tier={1: [True]},
            iterations=[],
        )
        assert "Insufficient data" in text
