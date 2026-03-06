"""
ConvergenceAnalyzer — publication-quality charts from MetricsTracker output.

Generates 7 figures for the Self-Evolving Skill Graph evaluation:
  1-5: Per-metric trend plots (raw episode + iteration aggregate)
  6:   Combined 2×3 dashboard
  7:   Ablation comparison bar chart

Usage:
    from experiments.convergence_analyzer import ConvergenceAnalyzer

    ca = ConvergenceAnalyzer("experiments/results/run_001/metrics.csv")
    ca.plot_dashboard("dashboard.png")
    ca.plot_ablation("ablation_comparison.csv", "ablation_comparison.png")

Chart style: seaborn aesthetic, fontsize=12, suitable for papers.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ── Style constants ──────────────────────────────────────────────────
_FONT_SIZE = 12
_TITLE_SIZE = 14
_LINE_ALPHA_RAW = 0.35
_LINE_ALPHA_AGG = 1.0
_LINE_WIDTH_RAW = 1.0
_LINE_WIDTH_AGG = 2.5
_FIG_SIZE_SINGLE = (10, 5)
_FIG_SIZE_DASHBOARD = (18, 12)
_FIG_SIZE_ABLATION = (14, 8)

# Metric display info: (label, ylabel, theory annotation, color)
_METRIC_INFO: Dict[str, Dict[str, str]] = {
    "rho": {
        "label": "ρ (Trace Compression Ratio)",
        "ylabel": "ρ",
        "theory": "理論預測：ρ↑",
        "color": "#2196F3",
        "direction": "up",
    },
    "kappa": {
        "label": "κ (Graph Contraction Rate)",
        "ylabel": "κ",
        "theory": "理論預測：κ→0",
        "color": "#FF9800",
        "direction": "down",
    },
    "entropy": {
        "label": "H(Gₜ) (Structural Entropy)",
        "ylabel": "H(Gₜ)",
        "theory": "理論預測：H→H*",
        "color": "#4CAF50",
        "direction": "converge",
    },
    "delta_sigma": {
        "label": "|Σₜ| and ΔΣ (Capacity Equilibrium)",
        "ylabel": "|Σₜ|",
        "theory": "理論預測：ΔΣ→0",
        "color": "#9C27B0",
        "direction": "zero",
    },
    "planning_depth": {
        "label": "E[Dₜ] (Planning Depth)",
        "ylabel": "E[D]",
        "theory": "理論預測：E[D]↓",
        "color": "#F44336",
        "direction": "down",
    },
}


# ═════════════════════════════════════════════════════════════════════
#  ConvergenceAnalyzer
# ═════════════════════════════════════════════════════════════════════

class ConvergenceAnalyzer:
    """Analyse and visualise MetricsTracker output.

    Args:
        metrics_csv: Path to the metrics CSV exported by MetricsTracker.
    """

    def __init__(self, metrics_csv: str) -> None:
        self._path = metrics_csv
        self._data: List[Dict[str, Any]] = self._load_csv(metrics_csv)
        logger.info(
            "Loaded %d episodes from %s", len(self._data), metrics_csv,
        )

    # ── Data loading ─────────────────────────────────────────────────

    @staticmethod
    def _load_csv(path: str) -> List[Dict[str, Any]]:
        """Load metrics CSV, casting numeric columns."""
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                parsed: Dict[str, Any] = {}
                for k, v in row.items():
                    if k == "timestamp":
                        parsed[k] = v
                    elif k in ("episode_id", "delta_sigma", "sigma_size"):
                        parsed[k] = int(v)
                    else:
                        parsed[k] = float(v)
                rows.append(parsed)
        return rows

    @property
    def episodes(self) -> List[int]:
        """Episode IDs in order."""
        return [r["episode_id"] for r in self._data]

    @property
    def num_episodes(self) -> int:
        return len(self._data)

    def _get_series(self, metric: str) -> List[float]:
        """Extract a metric as a flat list of floats."""
        return [float(r[metric]) for r in self._data]

    def _aggregate_iterations(
        self,
        metric: str,
        episodes_per_iteration: int = 10,
    ) -> Tuple[List[float], List[float]]:
        """Return (x_positions, y_values) for iteration-level aggregates.

        x_positions are the midpoint episode of each iteration window.
        """
        values = self._get_series(metric)
        eps = self.episodes
        n = max(1, episodes_per_iteration)
        xs, ys = [], []

        for i in range(0, len(values), n):
            chunk = values[i : i + n]
            chunk_eps = eps[i : i + n]
            xs.append(sum(chunk_eps) / len(chunk_eps))  # midpoint
            # For entropy and sigma_size, use last value; for others, mean
            if metric in ("entropy", "sigma_size"):
                ys.append(chunk[-1])
            else:
                ys.append(mean(chunk))

        return xs, ys

    # ── Convergence detection ────────────────────────────────────────

    def detect_convergence(
        self,
        metric: str = "entropy",
        window: int = 5,
        threshold: float = 0.05,
    ) -> Optional[int]:
        """Find convergence point for a metric.

        Returns the episode index where |Δmetric| stays below threshold for
        *window* consecutive episodes, or None if not reached.
        """
        values = self._get_series(metric)
        if len(values) < window + 1:
            return None

        consecutive = 0
        for i in range(1, len(values)):
            delta = abs(values[i] - values[i - 1])
            if delta < threshold:
                consecutive += 1
                if consecutive >= window:
                    return self.episodes[i - window + 1]
            else:
                consecutive = 0

        return None

    # ── Theoretical verification ─────────────────────────────────────

    def theoretical_verification(self) -> Dict[str, bool]:
        """Verify the 5 theoretical predictions.

        Returns a dict mapping metric name → boolean (True = pass).
        """
        if len(self._data) < 4:
            return {m: False for m in _METRIC_INFO}

        mid = len(self._data) // 2
        first_half = self._data[:mid]
        second_half = self._data[mid:]

        def _avg(records, key):
            return mean(float(r[key]) for r in records)

        results: Dict[str, bool] = {}

        # ρ should increase
        results["rho"] = _avg(second_half, "rho") >= _avg(first_half, "rho")

        # κ should approach 0
        results["kappa"] = (
            _avg(second_half, "kappa") <= _avg(first_half, "kappa")
            or _avg(second_half, "kappa") < 0.1
        )

        # H should converge (small change in late episodes)
        late = self._data[-max(5, len(self._data) // 5):]
        h_values = [r["entropy"] for r in late]
        results["entropy"] = (max(h_values) - min(h_values)) < 0.5

        # ΔΣ should approach 0
        late_deltas = [abs(r["delta_sigma"]) for r in self._data[-10:]]
        results["delta_sigma"] = max(late_deltas) <= 1

        # E[D] should decrease
        results["planning_depth"] = (
            _avg(second_half, "planning_depth")
            <= _avg(first_half, "planning_depth")
        )

        return results

    # ── Single metric plot ───────────────────────────────────────────

    def plot_single_metric(
        self,
        metric: str,
        output_path: str,
        episodes_per_iteration: int = 10,
    ) -> None:
        """Plot a single metric with raw episodes + iteration aggregates.

        Args:
            metric: One of 'rho', 'kappa', 'entropy', 'delta_sigma',
                    'planning_depth'.
            output_path: Where to save the PNG.
            episodes_per_iteration: Grouping size for aggregation.
        """
        info = _METRIC_INFO.get(metric, {})
        color = info.get("color", "#333333")

        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)

        eps = self.episodes
        raw = self._get_series(metric)

        # Special handling for capacity: dual y-axis
        if metric == "delta_sigma":
            self._plot_capacity(ax, eps, episodes_per_iteration, output_path)
            plt.close(fig)
            return

        # Raw per-episode (light)
        ax.plot(
            eps, raw,
            color=color, alpha=_LINE_ALPHA_RAW,
            linewidth=_LINE_WIDTH_RAW, label="Per-episode",
        )

        # Iteration aggregate (bold)
        agg_x, agg_y = self._aggregate_iterations(
            metric, episodes_per_iteration,
        )
        ax.plot(
            agg_x, agg_y,
            color=color, alpha=_LINE_ALPHA_AGG,
            linewidth=_LINE_WIDTH_AGG, label="Iteration avg",
            marker="o", markersize=5,
        )

        # Entropy-specific: log(K) upper bound + H* convergence line
        if metric == "entropy":
            sigma_sizes = self._get_series("sigma_size")
            max_k = max(sigma_sizes) if sigma_sizes else 1
            if max_k > 0:
                log_k = math.log(max(max_k, 1))
                ax.axhline(
                    y=log_k, color="gray", linestyle="--",
                    alpha=0.6, label=f"ln(K) = {log_k:.2f}",
                )
            # H* convergence line (average of last 20% episodes)
            tail = raw[max(0, len(raw) - len(raw) // 5):]
            if tail:
                h_star = mean(tail)
                ax.axhline(
                    y=h_star, color="#66BB6A", linestyle=":",
                    alpha=0.7, label=f"H* ≈ {h_star:.2f}",
                )

        # Theory annotation
        theory = info.get("theory", "")
        direction = info.get("direction", "")
        verification = self.theoretical_verification()
        passed = verification.get(metric, False)
        arrow_color = "#4CAF50" if passed else "#F44336"

        if theory:
            ax.annotate(
                theory,
                xy=(0.98, 0.95), xycoords="axes fraction",
                fontsize=11, ha="right", va="top",
                color=arrow_color,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=arrow_color, alpha=0.15,
                ),
            )

        # Mark convergence point
        conv = self.detect_convergence(metric)
        if conv is not None and metric in ("entropy", "kappa"):
            ax.axvline(
                x=conv, color="#FF5722", linestyle="--",
                alpha=0.5, label=f"Convergence @ ep {conv}",
            )

        ax.set_xlabel("Episode", fontsize=_FONT_SIZE)
        ax.set_ylabel(info.get("ylabel", metric), fontsize=_FONT_SIZE)
        ax.set_title(info.get("label", metric), fontsize=_TITLE_SIZE)
        ax.legend(fontsize=10, loc="best")
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s → %s", metric, output_path)

    def _plot_capacity(
        self,
        ax_unused: plt.Axes,
        eps: List[int],
        episodes_per_iteration: int,
        output_path: str,
    ) -> None:
        """Dual y-axis plot for |Σ| and ΔΣ."""
        info = _METRIC_INFO["delta_sigma"]
        color_sigma = info["color"]
        color_delta = "#E91E63"

        plt.style.use("seaborn-v0_8")
        fig, ax1 = plt.subplots(figsize=_FIG_SIZE_SINGLE)

        sigma = self._get_series("sigma_size")
        delta = self._get_series("delta_sigma")

        # Left y-axis: |Σ|
        ax1.plot(
            eps, sigma,
            color=color_sigma, alpha=_LINE_ALPHA_AGG,
            linewidth=_LINE_WIDTH_AGG, label="|Σₜ| (graph size)",
        )
        ax1.set_xlabel("Episode", fontsize=_FONT_SIZE)
        ax1.set_ylabel("|Σₜ|", fontsize=_FONT_SIZE, color=color_sigma)
        ax1.tick_params(axis="y", labelcolor=color_sigma, labelsize=10)
        ax1.tick_params(axis="x", labelsize=10)

        # Right y-axis: ΔΣ
        ax2 = ax1.twinx()
        ax2.bar(
            eps, delta,
            color=color_delta, alpha=0.4,
            width=0.8, label="ΔΣ (change)",
        )
        ax2.set_ylabel("ΔΣ", fontsize=_FONT_SIZE, color=color_delta)
        ax2.tick_params(axis="y", labelcolor=color_delta, labelsize=10)
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        # Theory annotation
        verification = self.theoretical_verification()
        passed = verification.get("delta_sigma", False)
        arrow_color = "#4CAF50" if passed else "#F44336"
        ax1.annotate(
            info["theory"],
            xy=(0.98, 0.95), xycoords="axes fraction",
            fontsize=11, ha="right", va="top",
            color=arrow_color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=arrow_color, alpha=0.15,
            ),
        )

        ax1.set_title(info["label"], fontsize=_TITLE_SIZE)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="best")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved capacity → %s", output_path)

    # ── Dashboard ────────────────────────────────────────────────────

    def plot_dashboard(
        self,
        output_path: str,
        episodes_per_iteration: int = 10,
    ) -> None:
        """Generate a 2×3 combined dashboard.

        Panels: rho, kappa, entropy, capacity, planning_depth, gini.
        """
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 3, figsize=_FIG_SIZE_DASHBOARD)
        flat_axes = axes.flatten()

        metrics = ["rho", "kappa", "entropy", "delta_sigma", "planning_depth"]

        for i, metric in enumerate(metrics):
            ax = flat_axes[i]
            info = _METRIC_INFO[metric]
            color = info["color"]
            eps = self.episodes

            if metric == "delta_sigma":
                # Simplified: just plot |Σ| on dashboard
                sigma = self._get_series("sigma_size")
                ax.plot(
                    eps, sigma,
                    color=color, linewidth=_LINE_WIDTH_AGG,
                )
                ax.set_ylabel("|Σₜ|", fontsize=10)
            else:
                raw = self._get_series(metric)
                ax.plot(
                    eps, raw,
                    color=color, alpha=_LINE_ALPHA_RAW,
                    linewidth=_LINE_WIDTH_RAW,
                )
                agg_x, agg_y = self._aggregate_iterations(
                    metric, episodes_per_iteration,
                )
                ax.plot(
                    agg_x, agg_y,
                    color=color, linewidth=_LINE_WIDTH_AGG,
                    marker="o", markersize=3,
                )
                ax.set_ylabel(info["ylabel"], fontsize=10)

            ax.set_xlabel("Episode", fontsize=10)
            ax.set_title(info["label"], fontsize=11)
            ax.tick_params(labelsize=9)

            # Theory pass/fail badge
            verification = self.theoretical_verification()
            passed = verification.get(metric, False)
            badge = "✓ PASS" if passed else "✗ FAIL"
            badge_color = "#4CAF50" if passed else "#F44336"
            ax.text(
                0.02, 0.95, badge,
                transform=ax.transAxes, fontsize=10,
                va="top", ha="left", color=badge_color,
                fontweight="bold",
            )

        # Panel 6: Gini coefficient over time (if sigma_size > 0)
        ax_gini = flat_axes[5]
        self._plot_gini_panel(ax_gini)

        fig.suptitle(
            "Self-Evolving Skill Graph — Metrics Dashboard",
            fontsize=16, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved dashboard → %s", output_path)

    def _plot_gini_panel(self, ax: plt.Axes) -> None:
        """Plot Gini coefficient in the 6th panel (if data available)."""
        # Compute Gini from sigma_size ratios (proxy)
        # Since we only have CSV data (no utility access), use sigma_size
        # as a proxy to show growth trajectory
        sigma = self._get_series("sigma_size")
        eps = self.episodes

        if not sigma or max(sigma) == 0:
            ax.text(
                0.5, 0.5, "No graph data",
                transform=ax.transAxes, fontsize=12,
                ha="center", va="center", color="gray",
            )
            ax.set_title("Gini Coefficient", fontsize=11)
            return

        # Use rolling entropy / ln(K) ratio as a structure proxy
        entropies = self._get_series("entropy")
        ratios = []
        for e, s in zip(entropies, sigma):
            if s > 1:
                # Normalized entropy: H / ln(|Σ|) — 1.0 = uniform, <1 = unequal
                ratios.append(1.0 - e / math.log(s) if e > 0 else 1.0)
            else:
                ratios.append(0.0)

        ax.plot(
            eps, ratios,
            color="#795548", linewidth=_LINE_WIDTH_AGG,
        )
        ax.fill_between(eps, ratios, alpha=0.2, color="#795548")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Structure Index", fontsize=10)
        ax.set_title("Core-Periphery Structure (1 − H/ln|Σ|)", fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(labelsize=9)

    # ── Ablation comparison ──────────────────────────────────────────

    def plot_ablation(
        self,
        ablation_csv: str,
        output_path: str,
    ) -> None:
        """Bar chart comparing ablation configurations.

        Args:
            ablation_csv: Path to ablation_comparison.csv.
            output_path: Where to save the PNG.
        """
        # Load ablation data
        rows: List[Dict[str, Any]] = []
        with open(ablation_csv, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if k == "config":
                        parsed[k] = v
                    elif k in ("final_sigma_size", "convergence_episode"):
                        parsed[k] = int(v)
                    else:
                        parsed[k] = float(v)
                rows.append(parsed)

        if not rows:
            logger.warning("No ablation data to plot")
            return

        plt.style.use("seaborn-v0_8")

        configs = [r["config"] for r in rows]
        metrics_to_plot = [
            ("final_rho", "ρ", "#2196F3"),
            ("final_kappa", "κ", "#FF9800"),
            ("final_entropy", "H", "#4CAF50"),
            ("final_planning_depth", "E[D]", "#F44336"),
            ("gini_coefficient", "Gini", "#795548"),
        ]

        fig, axes = plt.subplots(
            1, len(metrics_to_plot),
            figsize=_FIG_SIZE_ABLATION,
            sharey=False,
        )

        x = np.arange(len(configs))

        for i, (key, label, color) in enumerate(metrics_to_plot):
            ax = axes[i]
            values = [r.get(key, 0) for r in rows]

            bars = ax.bar(
                x, values, color=color, alpha=0.8,
                edgecolor="white", linewidth=0.5,
            )

            # Highlight full_system bar
            for j, conf in enumerate(configs):
                if conf == "full_system":
                    bars[j].set_edgecolor("#333333")
                    bars[j].set_linewidth(2)

            ax.set_title(label, fontsize=_FONT_SIZE, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(
                configs, rotation=45, ha="right", fontsize=8,
            )
            ax.tick_params(axis="y", labelsize=9)

            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(max(values, default=1), 0.1),
                    f"{val:.2f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", fontsize=7,
                )

        fig.suptitle(
            "Ablation Study — Component Contribution",
            fontsize=16, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved ablation comparison → %s", output_path)

    # ── Convenience: generate all plots ──────────────────────────────

    def generate_all(
        self,
        output_dir: str,
        episodes_per_iteration: int = 10,
        ablation_csv: Optional[str] = None,
    ) -> List[str]:
        """Generate all 7 figures, returning list of saved paths.

        Args:
            output_dir: Directory to save PNGs.
            episodes_per_iteration: Grouping size.
            ablation_csv: Optional path to ablation_comparison.csv.

        Returns:
            List of generated file paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: List[str] = []

        # Figures 1-5: individual metrics
        metric_files = {
            "rho": "rho_trend.png",
            "kappa": "kappa_trend.png",
            "entropy": "entropy_trend.png",
            "delta_sigma": "capacity_trend.png",
            "planning_depth": "depth_trend.png",
        }
        for metric, filename in metric_files.items():
            path = str(out / filename)
            self.plot_single_metric(
                metric, path, episodes_per_iteration,
            )
            saved.append(path)

        # Figure 6: dashboard
        dash_path = str(out / "dashboard.png")
        self.plot_dashboard(dash_path, episodes_per_iteration)
        saved.append(dash_path)

        # Figure 7: ablation (if data available)
        if ablation_csv and Path(ablation_csv).exists():
            abl_path = str(out / "ablation_comparison.png")
            self.plot_ablation(ablation_csv, abl_path)
            saved.append(abl_path)

        return saved

    def __repr__(self) -> str:
        return (
            f"ConvergenceAnalyzer(episodes={self.num_episodes}, "
            f"path={self._path!r})"
        )
