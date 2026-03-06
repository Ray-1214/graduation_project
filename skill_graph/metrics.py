"""
MetricsTracker — five theoretical evaluation criteria for the
Self-Evolving Skill Graph.

Tracks the structural invariants derived from the formal framework
(paper Section 7) to evaluate whether the system's evolution
matches theoretical predictions:

  1. Trace Compression Ratio  ρ   (Planning Depth Reduction)
  2. Graph Contraction Rate   κ   (Theorem 5 – Structural Equilibrium)
  3. Structural Entropy       H   (Proposition 3 – Entropy Convergence)
  4. Memory Capacity Δ|Σ|         (Theorem 5(i) – Size Convergence)
  5. Planning Depth           E[D] (Mean trace length)

References:
  - Self-Evolving Skill Graph, Section 7
  - Agent0-VL iteration-based reporting (arXiv:2511.19900)
"""

from __future__ import annotations

import csv
import json
import logging
import math
from datetime import datetime, timezone
from statistics import mean
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from skill_graph.skill_graph import SkillGraph

logger = logging.getLogger(__name__)


# ── Trend helpers ────────────────────────────────────────────────────

def _linear_slope(values: List[float]) -> float:
    """Compute the OLS slope of *values* against their indices.

    Returns 0.0 if fewer than 2 data points.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def _trend_symbol(slope: float, threshold: float = 1e-4) -> str:
    """Classify a slope as ↑ / ↓ / → (stable)."""
    if slope > threshold:
        return "↑"
    if slope < -threshold:
        return "↓"
    return "→"


# ═════════════════════════════════════════════════════════════════════
#  MetricsTracker
# ═════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """Track the five theoretical evaluation criteria across episodes.

    Args:
        contraction_window: Number of recent episodes used to compute
            the graph contraction rate κ.
    """

    def __init__(self, contraction_window: int = 10) -> None:
        self.contraction_window: int = max(1, contraction_window)
        self._history: List[Dict[str, Any]] = []
        self._prev_sigma_size: Optional[int] = None

    # ── Core recording ───────────────────────────────────────────────

    def record(
        self,
        episode_id: int,
        graph: "SkillGraph",
        raw_trace_lengths: List[int],
        compressed_trace_lengths: List[int],
        contraction_ops: int = 0,
        total_graph_ops: int = 0,
    ) -> Dict[str, Any]:
        """Record one episode and compute all five metrics.

        Args:
            episode_id:               Unique episode identifier.
            graph:                    Current SkillGraph state.
            raw_trace_lengths:        Trace lengths *without* skill macros.
            compressed_trace_lengths: Trace lengths *with* skill macros.
            contraction_ops:          Number of contraction (abstraction)
                                      operations in this episode.
            total_graph_ops:          Total graph operations in this episode.

        Returns:
            A dict containing the episode record.
        """
        # ── 1. Trace Compression Ratio  ρ ────────────────────────────
        raw_sum = sum(raw_trace_lengths)
        compressed_sum = sum(compressed_trace_lengths)
        rho = 1.0 - (compressed_sum / raw_sum) if raw_sum > 0 else 0.0

        # ── 2. Graph Contraction Rate  κ (sliding window) ────────────
        # Accumulate ops from this episode into the window
        window_start = max(0, len(self._history) - self.contraction_window + 1)
        window_records = self._history[window_start:]

        total_contract = contraction_ops + sum(
            r["_contraction_ops"] for r in window_records
        )
        total_ops = total_graph_ops + sum(
            r["_total_graph_ops"] for r in window_records
        )
        kappa = total_contract / total_ops if total_ops > 0 else 0.0

        # ── 3. Structural Entropy  H(G_t) ───────────────────────────
        entropy = graph.compute_entropy()

        # ── 4. Memory Capacity Equilibrium  ΔΣ ──────────────────────
        sigma_size = len(graph)
        if self._prev_sigma_size is None:
            delta_sigma = sigma_size          # first episode
        else:
            delta_sigma = sigma_size - self._prev_sigma_size
        self._prev_sigma_size = sigma_size

        # ── 5. Planning Depth  E[D] ─────────────────────────────────
        planning_depth = (
            mean(compressed_trace_lengths)
            if compressed_trace_lengths
            else 0.0
        )

        # ── Build record ─────────────────────────────────────────────
        record: Dict[str, Any] = {
            "episode_id": episode_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rho": rho,
            "kappa": kappa,
            "entropy": entropy,
            "delta_sigma": delta_sigma,
            "planning_depth": planning_depth,
            "sigma_size": sigma_size,
            # Internal bookkeeping (for κ sliding window)
            "_contraction_ops": contraction_ops,
            "_total_graph_ops": total_graph_ops,
        }
        self._history.append(record)

        logger.info(
            "Episode %d metrics: ρ=%.3f  κ=%.3f  H=%.3f  "
            "ΔΣ=%+d  E[D]=%.1f  |Σ|=%d",
            episode_id, rho, kappa, entropy,
            delta_sigma, planning_depth, sigma_size,
        )
        return record

    # ── Queries ───────────────────────────────────────────────────────

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all recorded episodes (including internal fields)."""
        return list(self._history)

    def get_iteration_summary(
        self,
        episodes_per_iteration: int = 10,
    ) -> List[Dict[str, Any]]:
        """Group episodes into fixed-size iterations and aggregate.

        Follows the Agent0-VL iteration-based reporting pattern
        (arXiv:2511.19900).

        Returns a list of dicts, each containing:
          - iteration:       0-based iteration index
          - episodes:        (start_id, end_id) inclusive
          - mean_rho:        Average trace compression ratio
          - mean_kappa:      Average contraction rate
          - last_entropy:    Entropy at the end of the iteration
          - last_sigma_size: |Σ| at the end of the iteration
          - mean_depth:      Average planning depth
        """
        if not self._history:
            return []

        n = max(1, episodes_per_iteration)
        summaries: List[Dict[str, Any]] = []

        for i in range(0, len(self._history), n):
            chunk = self._history[i : i + n]
            summaries.append({
                "iteration": i // n,
                "episodes": (
                    chunk[0]["episode_id"],
                    chunk[-1]["episode_id"],
                ),
                "mean_rho": mean(r["rho"] for r in chunk),
                "mean_kappa": mean(r["kappa"] for r in chunk),
                "last_entropy": chunk[-1]["entropy"],
                "last_sigma_size": chunk[-1]["sigma_size"],
                "mean_depth": mean(r["planning_depth"] for r in chunk),
            })

        return summaries

    # ── Export ─────────────────────────────────────────────────────────

    _CSV_COLUMNS = [
        "episode_id", "timestamp", "rho", "kappa",
        "entropy", "delta_sigma", "planning_depth", "sigma_size",
    ]

    def export_csv(self, filepath: str) -> None:
        """Write all recorded metrics to a CSV file.

        Columns: episode_id, timestamp, rho, kappa, entropy,
                 delta_sigma, planning_depth, sigma_size.
        """
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=self._CSV_COLUMNS,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(self._history)

        logger.info("Exported %d records to %s", len(self._history), filepath)

    def export_json(self, filepath: str) -> None:
        """Write all recorded metrics to a JSON file.

        Internal bookkeeping fields (prefixed with ``_``) are excluded.
        """
        clean = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in self._history
        ]
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2, ensure_ascii=False)

        logger.info("Exported %d records to %s", len(self._history), filepath)

    # ── Summary report ────────────────────────────────────────────────

    def summary(self) -> str:
        """Generate a text summary with trend analysis.

        Includes:
          - Per-metric trend direction (↑ / ↓ / →)
          - Theory compliance check (ρ↑, κ→0, H→H*, ΔΣ→0, E[D]↓)
          - Iteration-level aggregation table
        """
        if not self._history:
            return "No metrics recorded yet."

        n = len(self._history)
        lines: List[str] = [
            f"═══ Metrics Summary ({n} episodes) ═══",
            "",
        ]

        # ── Trend analysis ───────────────────────────────────────────
        rhos = [r["rho"] for r in self._history]
        kappas = [r["kappa"] for r in self._history]
        entropies = [r["entropy"] for r in self._history]
        deltas = [r["delta_sigma"] for r in self._history]
        depths = [r["planning_depth"] for r in self._history]

        slope_rho = _linear_slope(rhos)
        slope_kappa = _linear_slope(kappas)
        slope_entropy = _linear_slope(entropies)
        slope_delta = _linear_slope(deltas)
        slope_depth = _linear_slope(depths)

        trend_rho = _trend_symbol(slope_rho)
        trend_kappa = _trend_symbol(slope_kappa)
        trend_entropy = _trend_symbol(slope_entropy)
        trend_delta = _trend_symbol(slope_delta)
        trend_depth = _trend_symbol(slope_depth)

        # Latest values
        latest = self._history[-1]

        lines.append("Metric Trends:")
        lines.append(f"  ρ  (Trace Compression)    {trend_rho}  "
                      f"latest={latest['rho']:.4f}  slope={slope_rho:+.6f}")
        lines.append(f"  κ  (Contraction Rate)     {trend_kappa}  "
                      f"latest={latest['kappa']:.4f}  slope={slope_kappa:+.6f}")
        lines.append(f"  H  (Structural Entropy)   {trend_entropy}  "
                      f"latest={latest['entropy']:.4f}  slope={slope_entropy:+.6f}")
        lines.append(f"  ΔΣ (Capacity Change)      {trend_delta}  "
                      f"latest={latest['delta_sigma']:+d}  slope={slope_delta:+.6f}")
        lines.append(f"  E[D] (Planning Depth)     {trend_depth}  "
                      f"latest={latest['planning_depth']:.2f}  slope={slope_depth:+.6f}")
        lines.append("")

        # ── Theory compliance ────────────────────────────────────────
        lines.append("Theory Compliance:")

        checks = [
            ("ρ should increase (↑)",
             slope_rho > 0 or (n < 3),
             "ρ↑" if slope_rho > 0 else "ρ→" if abs(slope_rho) < 1e-4 else "ρ↓"),

            ("κ should approach 0 (→0)",
             slope_kappa <= 0 or latest["kappa"] < 0.1,
             f"κ={latest['kappa']:.3f}"),

            ("H should converge (→H*)",
             n < 5 or abs(slope_entropy) < 0.05,
             f"ΔH slope={slope_entropy:+.4f}"),

            ("ΔΣ should approach 0 (→0)",
             n < 3 or abs(latest["delta_sigma"]) <= 1,
             f"ΔΣ={latest['delta_sigma']:+d}"),

            ("E[D] should decrease (↓)",
             slope_depth <= 0 or (n < 3),
             f"E[D] slope={slope_depth:+.4f}"),
        ]

        for description, ok, detail in checks:
            status = "✓" if ok else "✗"
            lines.append(f"  [{status}] {description}  ({detail})")

        lines.append("")

        # ── Iteration summary table ──────────────────────────────────
        iterations = self.get_iteration_summary()
        if iterations:
            lines.append("Iteration Summary:")
            lines.append(
                f"  {'Iter':>4}  {'Episodes':>12}  "
                f"{'ρ':>7}  {'κ':>7}  {'H':>7}  "
                f"{'|Σ|':>5}  {'E[D]':>7}"
            )
            lines.append("  " + "─" * 60)
            for it in iterations:
                ep_range = f"{it['episodes'][0]}-{it['episodes'][1]}"
                lines.append(
                    f"  {it['iteration']:4d}  {ep_range:>12}  "
                    f"{it['mean_rho']:7.4f}  {it['mean_kappa']:7.4f}  "
                    f"{it['last_entropy']:7.4f}  "
                    f"{it['last_sigma_size']:5d}  "
                    f"{it['mean_depth']:7.2f}"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MetricsTracker(episodes={len(self._history)})"
