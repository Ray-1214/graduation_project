"""
Tests for ConvergenceAnalyzer.

Uses synthetic CSV data to test:
  - CSV loading and properties
  - Convergence detection
  - Theoretical verification
  - Plot generation (file existence, no crashes)
  - Ablation comparison chart
  - generate_all convenience method
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from experiments.convergence_analyzer import ConvergenceAnalyzer


# ── Fixture: create a synthetic metrics CSV ──────────────────────────

def _make_csv(path: Path, n: int = 30, evolving: bool = True) -> str:
    """Write a synthetic metrics CSV and return its path.

    If evolving=True, metrics follow theoretical predictions:
      ρ increases, κ decreases, H converges, ΔΣ→0, E[D] decreases.
    """
    path.mkdir(parents=True, exist_ok=True)
    filepath = str(path / "metrics.csv")
    fields = [
        "episode_id", "timestamp", "rho", "kappa",
        "entropy", "delta_sigma", "planning_depth", "sigma_size",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            t = i / max(n - 1, 1)  # 0..1 progress
            if evolving:
                rho = 0.35 * t
                kappa = 0.4 * (1 - t)
                entropy = 2.0 + 1.0 * (1 - math.exp(-3 * t))  # saturates ~3
                sigma_size = int(3 + 9 * t)
                delta_sigma = 1 if i < n // 3 else 0
                depth = 12 - 5 * t
            else:
                rho = 0.0
                kappa = 0.0
                entropy = 0.0
                sigma_size = 0
                delta_sigma = 0
                depth = 10.0

            writer.writerow({
                "episode_id": i,
                "timestamp": f"2026-01-01T00:{i:02d}:00Z",
                "rho": f"{rho:.4f}",
                "kappa": f"{kappa:.4f}",
                "entropy": f"{entropy:.4f}",
                "delta_sigma": delta_sigma,
                "planning_depth": f"{depth:.2f}",
                "sigma_size": sigma_size,
            })
    return filepath


def _make_ablation_csv(path: Path) -> str:
    """Write a synthetic ablation_comparison.csv."""
    filepath = str(path / "ablation_comparison.csv")
    fields = [
        "config", "final_rho", "final_kappa", "final_entropy",
        "final_sigma_size", "final_planning_depth",
        "convergence_episode", "gini_coefficient",
    ]
    rows = [
        {
            "config": "full_system",
            "final_rho": "0.355",
            "final_kappa": "0.010",
            "final_entropy": "3.09",
            "final_sigma_size": "12",
            "final_planning_depth": "7.0",
            "convergence_episode": "35",
            "gini_coefficient": "0.62",
        },
        {
            "config": "no_contraction",
            "final_rho": "0.180",
            "final_kappa": "0.000",
            "final_entropy": "3.50",
            "final_sigma_size": "25",
            "final_planning_depth": "9.5",
            "convergence_episode": "-1",
            "gini_coefficient": "0.30",
        },
        {
            "config": "vanilla_baseline",
            "final_rho": "0.000",
            "final_kappa": "0.000",
            "final_entropy": "0.00",
            "final_sigma_size": "0",
            "final_planning_depth": "12.0",
            "convergence_episode": "-1",
            "gini_coefficient": "0.00",
        },
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return filepath


@pytest.fixture
def csv_path(tmp_path):
    return _make_csv(tmp_path)


@pytest.fixture
def analyzer(csv_path):
    return ConvergenceAnalyzer(csv_path)


# ═══════════════════════════════════════════════════════════════════
#  Loading and properties
# ═══════════════════════════════════════════════════════════════════

class TestLoading:

    def test_loads_correct_count(self, analyzer):
        assert analyzer.num_episodes == 30

    def test_episodes_list(self, analyzer):
        assert analyzer.episodes == list(range(30))

    def test_repr(self, analyzer):
        r = repr(analyzer)
        assert "ConvergenceAnalyzer" in r
        assert "30" in r


# ═══════════════════════════════════════════════════════════════════
#  Convergence detection
# ═══════════════════════════════════════════════════════════════════

class TestConvergence:

    def test_detects_entropy_convergence(self, analyzer):
        # Entropy saturates → should converge
        ep = analyzer.detect_convergence("entropy", window=5, threshold=0.1)
        assert ep is not None
        assert ep > 0

    def test_rho_may_not_converge(self, analyzer):
        # rho keeps increasing → may not converge with tight threshold
        ep = analyzer.detect_convergence("rho", window=5, threshold=0.001)
        # Could be None or a late episode
        # Just ensure no crash
        assert ep is None or isinstance(ep, int)

    def test_insufficient_data(self, tmp_path):
        path = _make_csv(tmp_path / "short", n=3)
        ca = ConvergenceAnalyzer(path)
        ep = ca.detect_convergence("entropy", window=5)
        assert ep is None

    def test_flat_data_converges(self, tmp_path):
        path = _make_csv(tmp_path / "flat", n=20, evolving=False)
        ca = ConvergenceAnalyzer(path)
        ep = ca.detect_convergence("entropy", window=5, threshold=0.05)
        assert ep is not None  # constant → converges immediately


# ═══════════════════════════════════════════════════════════════════
#  Theoretical verification
# ═══════════════════════════════════════════════════════════════════

class TestVerification:

    def test_all_pass_evolving(self, analyzer):
        results = analyzer.theoretical_verification()
        assert isinstance(results, dict)
        assert len(results) == 5
        # With evolving data, most should pass
        assert results["rho"] is True  # ρ increases
        assert results["kappa"] is True  # κ decreases
        assert results["planning_depth"] is True  # E[D] decreases

    def test_insufficient_data_all_fail(self, tmp_path):
        path = _make_csv(tmp_path / "tiny", n=2)
        ca = ConvergenceAnalyzer(path)
        results = ca.theoretical_verification()
        assert all(v is False for v in results.values())


# ═══════════════════════════════════════════════════════════════════
#  Plot generation (smoke tests)
# ═══════════════════════════════════════════════════════════════════

class TestPlotSingleMetric:

    @pytest.mark.parametrize("metric", [
        "rho", "kappa", "entropy", "delta_sigma", "planning_depth",
    ])
    def test_creates_file(self, analyzer, tmp_path, metric):
        out = str(tmp_path / f"{metric}.png")
        analyzer.plot_single_metric(metric, out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_custom_episodes_per_iter(self, analyzer, tmp_path):
        out = str(tmp_path / "rho_5.png")
        analyzer.plot_single_metric("rho", out, episodes_per_iteration=5)
        assert Path(out).exists()


class TestPlotDashboard:

    def test_creates_file(self, analyzer, tmp_path):
        out = str(tmp_path / "dashboard.png")
        analyzer.plot_dashboard(out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 1000  # non-trivial

    def test_flat_data(self, tmp_path):
        path = _make_csv(tmp_path / "flat", n=10, evolving=False)
        ca = ConvergenceAnalyzer(path)
        out = str(tmp_path / "dashboard_flat.png")
        ca.plot_dashboard(out)
        assert Path(out).exists()


class TestPlotAblation:

    def test_creates_file(self, analyzer, tmp_path):
        abl = _make_ablation_csv(tmp_path)
        out = str(tmp_path / "ablation.png")
        analyzer.plot_ablation(abl, out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 1000


# ═══════════════════════════════════════════════════════════════════
#  generate_all convenience
# ═══════════════════════════════════════════════════════════════════

class TestGenerateAll:

    def test_generates_six_without_ablation(self, analyzer, tmp_path):
        out = str(tmp_path / "all_plots")
        saved = analyzer.generate_all(out)
        assert len(saved) == 6  # 5 metrics + dashboard (no ablation)
        for path in saved:
            assert Path(path).exists()

    def test_generates_seven_with_ablation(self, analyzer, tmp_path):
        abl = _make_ablation_csv(tmp_path)
        out = str(tmp_path / "all_plots_abl")
        saved = analyzer.generate_all(out, ablation_csv=abl)
        assert len(saved) == 7  # +ablation chart
        for path in saved:
            assert Path(path).exists()
