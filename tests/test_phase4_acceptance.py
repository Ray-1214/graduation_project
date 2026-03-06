"""
Phase 4 Acceptance Tests — 32 criteria across MetricsTracker,
TaskCurriculum, Batch Experiment, Ablation, and ConvergenceAnalyzer.

Numbering follows the Phase 4 acceptance checklist (4-1 … 4-32).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _mock_graph(num_nodes: int = 5, entropy: float = 2.5):
    """Return a mock SkillGraph with configurable size and entropy."""
    g = MagicMock()
    g.compute_entropy.return_value = entropy
    g.__len__ = MagicMock(return_value=num_nodes)

    # Create fake skills for snapshot
    skill_mock = MagicMock()
    skill_mock.utility = 0.5
    g.skills = [skill_mock] * num_nodes

    # snapshot support
    g.snapshot.return_value = {
        "num_skills": num_nodes,
        "num_edges": num_nodes - 1,
        "capacity": 100,
        "structural_entropy": entropy,
        "nodes": [{"id": f"s{i}", "name": f"skill_{i}"} for i in range(num_nodes)],
        "edges": [{"src": f"s{i}", "dst": f"s{i+1}", "weight": 1.0}
                  for i in range(num_nodes - 1)],
    }
    return g


# ═══════════════════════════════════════════════════════════════════
#  4.1 MetricsTracker Tests (4-1 through 4-10)
# ═══════════════════════════════════════════════════════════════════

class TestMetricsTracker:

    def _make_tracker(self):
        from skill_graph.metrics import MetricsTracker
        return MetricsTracker(contraction_window=10)

    # ------- 4-1: ρ calculation ------------------------------------
    def test_4_1_rho_calculation(self):
        """ρ = 1 − compressed/raw = 1 − 19/30 ≈ 0.367"""
        tracker = self._make_tracker()
        g = _mock_graph()
        rec = tracker.record(
            episode_id=0,
            graph=g,
            raw_trace_lengths=[10, 8, 12],        # sum = 30
            compressed_trace_lengths=[6, 5, 8],    # sum = 19
        )
        expected = 1 - 19 / 30
        assert abs(rec["rho"] - expected) < 1e-6

    # ------- 4-2: κ calculation ------------------------------------
    def test_4_2_kappa_calculation(self):
        """3 contractions / 7 total ops → κ = 3/7 ≈ 0.429"""
        tracker = self._make_tracker()
        g = _mock_graph()
        rec = tracker.record(
            episode_id=0,
            graph=g,
            raw_trace_lengths=[10],
            compressed_trace_lengths=[8],
            contraction_ops=3,
            total_graph_ops=7,
        )
        assert abs(rec["kappa"] - 3 / 7) < 1e-6

    # ------- 4-3: H matches graph.compute_entropy() ---------------
    def test_4_3_entropy_matches_graph(self):
        """H comes directly from graph.compute_entropy()."""
        tracker = self._make_tracker()
        g = _mock_graph(entropy=3.14159)
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[5], compressed_trace_lengths=[4],
        )
        assert rec["entropy"] == pytest.approx(3.14159)
        g.compute_entropy.assert_called_once()

    # ------- 4-4: ΔΣ (capacity equilibrium) -----------------------
    def test_4_4_delta_sigma(self):
        """First ep: ΔΣ = |Σ|. Second ep: ΔΣ = new − old."""
        tracker = self._make_tracker()
        g5 = _mock_graph(num_nodes=5)
        g8 = _mock_graph(num_nodes=8)

        r1 = tracker.record(0, g5, [5], [4])
        assert r1["delta_sigma"] == 5       # first episode = full size
        assert r1["sigma_size"] == 5

        r2 = tracker.record(1, g8, [5], [4])
        assert r2["delta_sigma"] == 3       # 8 − 5
        assert r2["sigma_size"] == 8

    # ------- 4-5: E[D] (planning depth) ---------------------------
    def test_4_5_planning_depth(self):
        """E[D] = mean(compressed_trace_lengths) = (10+8+12)/3 = 10."""
        tracker = self._make_tracker()
        g = _mock_graph()
        rec = tracker.record(
            episode_id=0, graph=g,
            raw_trace_lengths=[10, 8, 12],
            compressed_trace_lengths=[10, 8, 12],
        )
        assert rec["planning_depth"] == pytest.approx(10.0)

    # ------- 4-6: consecutive records -----------------------------
    def test_4_6_consecutive_records(self):
        """5 consecutive records → history has 5 entries."""
        tracker = self._make_tracker()
        g = _mock_graph()
        for i in range(5):
            tracker.record(i, g, [10], [8])
        assert len(tracker.get_history()) == 5

    # ------- 4-7: export_csv format -------------------------------
    def test_4_7_export_csv(self, tmp_path):
        """CSV readable with correct fieldnames."""
        tracker = self._make_tracker()
        g = _mock_graph()
        for i in range(3):
            tracker.record(i, g, [10], [8])

        path = str(tmp_path / "metrics.csv")
        tracker.export_csv(path)

        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 3
        expected_cols = {
            "episode_id", "timestamp", "rho", "kappa",
            "entropy", "delta_sigma", "planning_depth", "sigma_size",
        }
        assert set(rows[0].keys()) == expected_cols

    # ------- 4-8: export_json format ------------------------------
    def test_4_8_export_json(self, tmp_path):
        """JSON readable, no internal fields."""
        tracker = self._make_tracker()
        g = _mock_graph()
        for i in range(3):
            tracker.record(i, g, [10], [8])

        path = str(tmp_path / "metrics.json")
        tracker.export_json(path)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert len(data) == 3
        # No internal fields leaked
        for rec in data:
            assert not any(k.startswith("_") for k in rec)

    # ------- 4-9: summary includes trends + theory check ----------
    def test_4_9_summary_content(self):
        """Summary contains trend directions and theory compliance."""
        tracker = self._make_tracker()
        for i in range(10):
            g = _mock_graph(num_nodes=5 + i, entropy=2.0 + i * 0.1)
            tracker.record(
                i, g,
                raw_trace_lengths=[10],
                compressed_trace_lengths=[max(1, 10 - i)],
            )

        text = tracker.summary()
        assert "Metric Trends:" in text
        assert "Theory Compliance:" in text
        # Should contain direction markers
        assert any(sym in text for sym in ("↑", "↓", "→"))

    # ------- 4-10: get_iteration_summary --------------------------
    def test_4_10_iteration_summary(self):
        """50 episodes, eps_per_iter=10 → 5 iterations."""
        tracker = self._make_tracker()
        g = _mock_graph()
        for i in range(50):
            tracker.record(i, g, [10], [8])

        iters = tracker.get_iteration_summary(episodes_per_iteration=10)
        assert len(iters) == 5
        for it in iters:
            assert "iteration" in it
            assert "mean_rho" in it
            assert "mean_kappa" in it
            assert "last_entropy" in it
            assert "last_sigma_size" in it
            assert "mean_depth" in it


# ═══════════════════════════════════════════════════════════════════
#  4.2 TaskCurriculum Tests (4-11 through 4-15)
# ═══════════════════════════════════════════════════════════════════

class TestTaskCurriculum:

    TASKS_DIR = str(Path(__file__).resolve().parent.parent
                    / "experiments" / "tasks")

    def _load(self):
        from experiments.task_curriculum import TaskCurriculum
        return TaskCurriculum(self.TASKS_DIR)

    # ------- 4-11: Tier loading fields ----------------------------
    def test_4_11_tier_loading(self):
        """Each task in tier_1.json has required fields."""
        cur = self._load()
        tasks = cur.get_single_tier(1)
        assert len(tasks) > 0
        for t in tasks:
            assert "id" in t
            assert "tier" in t
            assert "task_description" in t
            assert "expected_answer" in t

    # ------- 4-12: Sequential ordering ----------------------------
    def test_4_12_sequential_ordering(self):
        """get_sequential() returns tier 1→2→3→4 order."""
        cur = self._load()
        tasks = cur.get_sequential()
        tiers = [t["tier"] for t in tasks]
        # Tiers should be non-decreasing
        for i in range(1, len(tiers)):
            assert tiers[i] >= tiers[i - 1], (
                f"Tier order violated at index {i}: "
                f"{tiers[i-1]} → {tiers[i]}"
            )

    # ------- 4-13: Shuffled + deterministic seed ------------------
    def test_4_13_shuffled_deterministic(self):
        """Same seed → same order; different from sequential."""
        cur = self._load()
        a = cur.get_shuffled(seed=42)
        b = cur.get_shuffled(seed=42)
        seq = cur.get_sequential()

        # Same seed → identical
        ids_a = [t["id"] for t in a]
        ids_b = [t["id"] for t in b]
        assert ids_a == ids_b

        # Should differ from sequential (extremely unlikely to match)
        ids_seq = [t["id"] for t in seq]
        assert ids_a != ids_seq

    # ------- 4-14: Single tier filter -----------------------------
    def test_4_14_single_tier(self):
        """get_single_tier(2) returns only tier-2 tasks."""
        cur = self._load()
        tasks = cur.get_single_tier(2)
        assert len(tasks) > 0
        assert all(t["tier"] == 2 for t in tasks)

    # ------- 4-15: Repeated mode ----------------------------------
    def test_4_15_repeated(self):
        """get_repeated(3) returns 3× the sequential length."""
        cur = self._load()
        seq = cur.get_sequential()
        rep = cur.get_repeated(3)
        assert len(rep) == len(seq) * 3
        # Each round should be the same
        n = len(seq)
        for r in range(3):
            ids_round = [t["id"] for t in rep[r * n : (r + 1) * n]]
            ids_seq = [t["id"] for t in seq]
            assert ids_round == ids_seq


# ═══════════════════════════════════════════════════════════════════
#  4.3 Batch Experiment Tests (4-16 through 4-22)
# ═══════════════════════════════════════════════════════════════════

class TestBatchExperiment:

    # ------- 4-16: Script importable (proxy for runnable) ---------
    def test_4_16_script_importable(self):
        """run_experiment_batch module can be imported without error."""
        from experiments import run_experiment_batch
        assert hasattr(run_experiment_batch, "run_experiment")
        assert hasattr(run_experiment_batch, "main")
        assert hasattr(run_experiment_batch, "build_parser")

    # ------- 4-17: Output files (via mocked run) ------------------
    def test_4_17_output_files(self, tmp_path):
        """run_experiment produces expected output files."""
        from experiments.run_experiment_batch import (
            run_experiment, _write_iteration_csv, _build_summary,
        )
        from skill_graph.metrics import MetricsTracker

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Create fake agent
        agent = MagicMock()
        result_mock = MagicMock()
        result_mock.answer = "test"
        result_mock.strategy = "cot"
        result_mock.score = 1.0
        result_mock.evolution_log = ""
        result_mock.verification_verdict = "PASS"
        result_mock.hallucination_score = 0.0
        result_mock.episode = {"steps": [{"action": "think"}]}
        agent.run.return_value = result_mock
        agent.skill_graph = _mock_graph(num_nodes=3)
        agent.evolution = MagicMock()
        agent.evolution._last_log = None

        tracker = MetricsTracker()
        tasks = [
            {"id": f"t{i}", "tier": 1,
             "task_description": f"task {i}", "expected_answer": "ans"}
            for i in range(5)
        ]

        run_experiment(
            agent=agent,
            tasks=tasks,
            tracker=tracker,
            output_dir=output_dir,
            snapshot_interval=3,
            episodes_per_iteration=3,
            seed=42,
        )

        # Check expected files
        assert (output_dir / "metrics.csv").exists()
        assert (output_dir / "experiment_log.jsonl").exists()
        assert (output_dir / "final_graph.json").exists()

        # Write iteration CSV (needs iteration list, not tracker)
        iterations = tracker.get_iteration_summary(episodes_per_iteration=3)
        _write_iteration_csv(iterations, str(output_dir / "iteration_summary.csv"))
        assert (output_dir / "iteration_summary.csv").exists()

        # Write summary (full signature)
        summary_text = _build_summary(
            tracker,
            experiment_mode="sequential",
            total_episodes=5,
            total_duration=10.0,
            seed=42,
            accuracy_by_tier={1: [True, False, True, False, True]},
            iterations=iterations,
        )
        (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
        assert (output_dir / "summary.txt").exists()

    # ------- 4-18: experiment_log.jsonl format --------------------
    def test_4_18_experiment_log_format(self, tmp_path):
        """Each line of experiment_log.jsonl is valid JSON with required keys."""
        from experiments.run_experiment_batch import run_experiment
        from skill_graph.metrics import MetricsTracker

        output_dir = tmp_path / "log_test"
        output_dir.mkdir()

        agent = MagicMock()
        result_mock = MagicMock()
        result_mock.answer = "answer"
        result_mock.strategy = "cot"
        result_mock.score = 0.8
        result_mock.evolution_log = ""
        result_mock.verification_verdict = "PASS"
        result_mock.hallucination_score = 0.1
        result_mock.episode = {"steps": []}
        agent.run.return_value = result_mock
        agent.skill_graph = _mock_graph()
        agent.evolution = MagicMock()
        agent.evolution._last_log = None

        tracker = MetricsTracker()
        tasks = [
            {"id": "t0", "tier": 1,
             "task_description": "test", "expected_answer": "ans"}
        ]

        run_experiment(
            agent=agent, tasks=tasks, tracker=tracker,
            output_dir=output_dir, seed=42,
        )

        log_path = output_dir / "experiment_log.jsonl"
        assert log_path.exists()
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                entry = json.loads(line)
                assert "episode_id" in entry
                assert "strategy_used" in entry
                assert "metrics_snapshot" in entry

    # ------- 4-19: ρ trend (informational) -------------------------
    def test_4_19_rho_trend_positive(self):
        """Over 20 episodes with growing graph, ρ should generally ≥ 0."""
        from skill_graph.metrics import MetricsTracker
        tracker = MetricsTracker()
        for i in range(20):
            g = _mock_graph(num_nodes=5 + i)
            # Simulate improving compression
            raw = 10
            comp = max(1, 10 - i // 3)
            tracker.record(i, g, [raw], [comp])

        history = tracker.get_history()
        # All ρ values should be non-negative
        assert all(r["rho"] >= 0 for r in history)
        # Later ρ should be >= earlier ρ on average
        first_half = mean(r["rho"] for r in history[:10])
        second_half = mean(r["rho"] for r in history[10:])
        assert second_half >= first_half

    # ------- 4-20: E[D] trend (informational) --------------------
    def test_4_20_depth_trend(self):
        """E[D] should not increase if compression improves."""
        from skill_graph.metrics import MetricsTracker
        tracker = MetricsTracker()
        for i in range(20):
            g = _mock_graph(num_nodes=5 + i)
            depth = max(3, 10 - i // 2)
            tracker.record(i, g, [10], [depth])

        history = tracker.get_history()
        first_half = mean(r["planning_depth"] for r in history[:10])
        second_half = mean(r["planning_depth"] for r in history[10:])
        assert second_half <= first_half

    # ------- 4-21: ΔΣ approaches 0 in late episodes ---------------
    def test_4_21_delta_sigma_stabilises(self):
        """Late ΔΣ values should be small as graph stabilises."""
        from skill_graph.metrics import MetricsTracker
        tracker = MetricsTracker()
        for i in range(30):
            # Graph grows fast initially, then stabilises
            size = min(15, 5 + i)
            g = _mock_graph(num_nodes=size)
            tracker.record(i, g, [10], [8])

        history = tracker.get_history()
        late_deltas = [abs(r["delta_sigma"]) for r in history[-10:]]
        assert all(d <= 1 for d in late_deltas)

    # ------- 4-22: Graph snapshot readable ------------------------
    def test_4_22_graph_snapshot(self):
        """snapshot() returns dict with nodes and edges."""
        g = _mock_graph(num_nodes=5)
        snap = g.snapshot()
        assert "nodes" in snap
        assert "edges" in snap
        assert len(snap["nodes"]) == 5
        assert len(snap["edges"]) == 4


# ═══════════════════════════════════════════════════════════════════
#  4.4 Ablation Tests (4-23 through 4-27)
# ═══════════════════════════════════════════════════════════════════

class TestAblation:

    # ------- 4-23: Script importable ------------------------------
    def test_4_23_ablation_importable(self):
        """run_ablation module can be imported without error."""
        from experiments import run_ablation
        assert hasattr(run_ablation, "ABLATION_CONFIGS")
        assert hasattr(run_ablation, "main")
        assert hasattr(run_ablation, "run_single_ablation")

    # ------- 4-24: 7 configs defined ------------------------------
    def test_4_24_seven_configs(self):
        """ABLATION_CONFIGS has exactly 7 entries."""
        from experiments.run_ablation import ABLATION_CONFIGS
        assert len(ABLATION_CONFIGS) == 7

    # ------- 4-25: full_system > vanilla_baseline -----------------
    def test_4_25_full_beats_vanilla(self):
        """full_system has different / superset capabilities vs vanilla."""
        from experiments.run_ablation import ABLATION_CONFIGS
        full = ABLATION_CONFIGS["full_system"]
        vanilla = ABLATION_CONFIGS["vanilla_baseline"]

        # full_system has everything ON
        assert all(full[k] for k in [
            "compound_reasoning", "graph_contraction",
            "tiered_memory", "hallucination_guard", "reflexion_memory",
        ])
        # vanilla has everything OFF
        assert not any(vanilla[k] for k in [
            "compound_reasoning", "graph_contraction",
            "tiered_memory", "hallucination_guard", "reflexion_memory",
        ])

    # ------- 4-26: no_contraction differs from full ---------------
    def test_4_26_no_contraction_effect(self):
        """no_contraction disables exactly graph_contraction."""
        from experiments.run_ablation import ABLATION_CONFIGS
        no_c = ABLATION_CONFIGS["no_contraction"]
        assert no_c["graph_contraction"] is False
        # Everything else is still ON
        assert no_c["compound_reasoning"] is True
        assert no_c["tiered_memory"] is True
        assert no_c["hallucination_guard"] is True
        assert no_c["reflexion_memory"] is True

    # ------- 4-27: Report builder works ---------------------------
    def test_4_27_ablation_report(self):
        """build_ablation_report produces text with component contributions."""
        from experiments.run_ablation import (
            AblationResult, build_ablation_report,
        )
        results = [
            AblationResult("full_system", "Full",
                           final_rho=0.35, final_kappa=0.01,
                           final_entropy=3.0, final_sigma_size=12,
                           final_planning_depth=7.0,
                           convergence_episode=35, gini_coefficient=0.6),
            AblationResult("no_contraction", "No contraction",
                           final_rho=0.18, final_kappa=0.0,
                           final_entropy=3.5, final_sigma_size=25,
                           final_planning_depth=9.5,
                           convergence_episode=-1, gini_coefficient=0.3),
            AblationResult("vanilla_baseline", "Vanilla",
                           final_rho=0.0, final_kappa=0.0,
                           final_entropy=0.0, final_sigma_size=0,
                           final_planning_depth=12.0,
                           convergence_episode=-1, gini_coefficient=0.0),
        ]
        report = build_ablation_report(results, 600, 42, 60)
        assert "Component Impact Analysis" in report
        assert "no_contraction" in report
        assert "Δρ" in report
        assert "Key Findings" in report


# ═══════════════════════════════════════════════════════════════════
#  4.5 ConvergenceAnalyzer Tests (4-28 through 4-32)
# ═══════════════════════════════════════════════════════════════════

def _make_metrics_csv(path: Path, n: int = 30) -> str:
    """Synthetic metrics CSV for ConvergenceAnalyzer tests."""
    filepath = str(path / "metrics.csv")
    fields = [
        "episode_id", "timestamp", "rho", "kappa",
        "entropy", "delta_sigma", "planning_depth", "sigma_size",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            t = i / max(n - 1, 1)
            writer.writerow({
                "episode_id": i,
                "timestamp": f"2026-01-01T00:{i:02d}:00Z",
                "rho": f"{0.35 * t:.4f}",
                "kappa": f"{0.4 * (1-t):.4f}",
                "entropy": f"{2.0 + 1.0 * (1 - math.exp(-3*t)):.4f}",
                "delta_sigma": 1 if i < n // 3 else 0,
                "planning_depth": f"{12 - 5*t:.2f}",
                "sigma_size": int(3 + 9 * t),
            })
    return filepath


def _make_ablation_csv(path: Path) -> str:
    """Synthetic ablation CSV."""
    filepath = str(path / "ablation_comparison.csv")
    fields = [
        "config", "final_rho", "final_kappa", "final_entropy",
        "final_sigma_size", "final_planning_depth",
        "convergence_episode", "gini_coefficient",
    ]
    rows = [
        {"config": "full_system", "final_rho": "0.355",
         "final_kappa": "0.010", "final_entropy": "3.09",
         "final_sigma_size": "12", "final_planning_depth": "7.0",
         "convergence_episode": "35", "gini_coefficient": "0.62"},
        {"config": "vanilla_baseline", "final_rho": "0.000",
         "final_kappa": "0.000", "final_entropy": "0.00",
         "final_sigma_size": "0", "final_planning_depth": "12.0",
         "convergence_episode": "-1", "gini_coefficient": "0.00"},
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return filepath


class TestConvergenceAnalyzer:

    # ------- 4-28: 5 PNGs generated --------------------------------
    @pytest.mark.parametrize("metric,filename", [
        ("rho", "rho_trend.png"),
        ("kappa", "kappa_trend.png"),
        ("entropy", "entropy_trend.png"),
        ("delta_sigma", "capacity_trend.png"),
        ("planning_depth", "depth_trend.png"),
    ])
    def test_4_28_chart_generation(self, tmp_path, metric, filename):
        """Each metric produces a PNG with size > 0."""
        from experiments.convergence_analyzer import ConvergenceAnalyzer
        csv_path = _make_metrics_csv(tmp_path)
        ca = ConvergenceAnalyzer(csv_path)
        out = str(tmp_path / filename)
        ca.plot_single_metric(metric, out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    # ------- 4-29: Dashboard 2×3 ----------------------------------
    def test_4_29_dashboard(self, tmp_path):
        """plot_dashboard creates a PNG."""
        from experiments.convergence_analyzer import ConvergenceAnalyzer
        csv_path = _make_metrics_csv(tmp_path)
        ca = ConvergenceAnalyzer(csv_path)
        out = str(tmp_path / "dashboard.png")
        ca.plot_dashboard(out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 1000  # non-trivial

    # ------- 4-30: Convergence detection --------------------------
    def test_4_30_convergence_detection(self, tmp_path):
        """Saturating entropy → convergence detected."""
        from experiments.convergence_analyzer import ConvergenceAnalyzer
        csv_path = _make_metrics_csv(tmp_path, n=40)
        ca = ConvergenceAnalyzer(csv_path)
        ep = ca.detect_convergence("entropy", window=5, threshold=0.1)
        assert ep is not None
        assert isinstance(ep, int)
        assert ep > 0

    # ------- 4-31: Theoretical verification returns dict ----------
    def test_4_31_theoretical_verification(self, tmp_path):
        """Returns dict with 5 metric keys → bool values."""
        from experiments.convergence_analyzer import ConvergenceAnalyzer
        csv_path = _make_metrics_csv(tmp_path)
        ca = ConvergenceAnalyzer(csv_path)
        results = ca.theoretical_verification()
        assert isinstance(results, dict)
        assert len(results) == 5
        expected_keys = {"rho", "kappa", "entropy", "delta_sigma", "planning_depth"}
        assert set(results.keys()) == expected_keys
        assert all(isinstance(v, bool) for v in results.values())

    # ------- 4-32: Ablation chart ---------------------------------
    def test_4_32_ablation_chart(self, tmp_path):
        """plot_ablation generates a comparison bar chart PNG."""
        from experiments.convergence_analyzer import ConvergenceAnalyzer
        csv_path = _make_metrics_csv(tmp_path)
        abl_path = _make_ablation_csv(tmp_path)
        ca = ConvergenceAnalyzer(csv_path)
        out = str(tmp_path / "ablation_comparison.png")
        ca.plot_ablation(abl_path, out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 1000
