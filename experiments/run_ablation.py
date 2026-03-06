#!/usr/bin/env python3
"""
run_ablation.py — ablation experiment for the Self-Evolving Skill Graph.

Compares the full system against variants with specific components
disabled, quantifying each component's contribution to the five
structural evaluation criteria.

Inspired by Agent0 ablation design (arXiv:2511.16043, 2511.19900)
comparing Absolute-Zero / R-Zero / Socratic-Zero variants.

Usage:
    python experiments/run_ablation.py \\
        --tasks-dir experiments/tasks \\
        --output-dir experiments/results/ablation_001

Each config runs the SAME task set (sequential, fixed seed) so that
differences are attributable solely to the disabled component.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.task_curriculum import TaskCurriculum
from experiments.run_experiment_batch import (
    run_experiment,
    _format_duration,
)
from skill_graph.metrics import MetricsTracker

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  Ablation configurations
# ═════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "full_system": {
        "description": "完整系統（所有元件啟用）",
        "compound_reasoning": True,
        "graph_contraction": True,
        "tiered_memory": True,
        "hallucination_guard": True,
        "reflexion_memory": True,
    },
    "no_contraction": {
        "description": "關閉 graph contraction（只增不縮）",
        "compound_reasoning": True,
        "graph_contraction": False,
        "tiered_memory": True,
        "hallucination_guard": True,
        "reflexion_memory": True,
    },
    "no_tiered_memory": {
        "description": "平坦記憶體（不分 active/cold/archive）",
        "compound_reasoning": True,
        "graph_contraction": True,
        "tiered_memory": False,
        "hallucination_guard": True,
        "reflexion_memory": True,
    },
    "no_compound": {
        "description": "只用 CoT（force_strategy='cot_only'）",
        "compound_reasoning": False,
        "graph_contraction": True,
        "tiered_memory": True,
        "hallucination_guard": True,
        "reflexion_memory": True,
    },
    "no_hallucination_guard": {
        "description": "跳過事實驗證",
        "compound_reasoning": True,
        "graph_contraction": True,
        "tiered_memory": True,
        "hallucination_guard": False,
        "reflexion_memory": True,
    },
    "no_reflexion": {
        "description": "跳過反思記憶寫入",
        "compound_reasoning": True,
        "graph_contraction": True,
        "tiered_memory": True,
        "hallucination_guard": True,
        "reflexion_memory": False,
    },
    "vanilla_baseline": {
        "description": "純 LLM，不用任何 skill/memory/evolution",
        "compound_reasoning": False,
        "graph_contraction": False,
        "tiered_memory": False,
        "hallucination_guard": False,
        "reflexion_memory": False,
    },
}


# ═════════════════════════════════════════════════════════════════════
#  Agent configuration helpers
# ═════════════════════════════════════════════════════════════════════

def configure_agent(agent: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ablation config to a MainAgent, returning runtime hints.

    Modifies the agent **in-place** and returns a dict of runtime
    hints (e.g. ``force_strategy``, ``do_reflect``) that the
    experiment loop should use.
    """
    from skill_graph.evolution_operator import EvolutionOperator

    hints: Dict[str, Any] = {
        "force_strategy": None,
        "do_reflect": True,
    }

    # ── Compound reasoning ───────────────────────────────────────
    if not config.get("compound_reasoning", True):
        hints["force_strategy"] = "cot_only"

    # ── Graph contraction ────────────────────────────────────────
    if not config.get("graph_contraction", True):
        # Replace contraction step with a no-op
        agent.evolution._step_subgraph_contraction = (
            lambda graph, trace, log: None
        )

    # ── Tiered memory ────────────────────────────────────────────
    if not config.get("tiered_memory", True):
        # Replace tier update step with a no-op
        agent.evolution._step_memory_tier_update = (
            lambda graph, partition, log: None
        )

    # ── Hallucination guard ──────────────────────────────────────
    if not config.get("hallucination_guard", True):
        # Replace verify() with a pass-through that always returns PASS
        from reasoning.compound_result import VerificationResult, VERDICT_PASS

        class _NoOpGuard:
            """Dummy guard that always passes."""
            def verify(self, *args, **kwargs) -> VerificationResult:
                return VerificationResult(
                    verdict=VERDICT_PASS,
                    hallucination_score=0.0,
                    claims=[],
                )
            def __repr__(self) -> str:
                return "NoOpHallucinationGuard()"

        agent.compound_reasoner._hallucination_guard = _NoOpGuard()

    # ── Reflexion memory ─────────────────────────────────────────
    if not config.get("reflexion_memory", True):
        hints["do_reflect"] = False

    # ── Vanilla baseline extras ──────────────────────────────────
    if (not config.get("compound_reasoning") and
            not config.get("graph_contraction") and
            not config.get("tiered_memory")):
        # Skip evolution entirely by making evolve() return empty log
        from skill_graph.evolution_operator import EvolutionLog
        agent.evolution.evolve = (
            lambda graph, trace, partition: EvolutionLog()
        )

    return hints


# ═════════════════════════════════════════════════════════════════════
#  Convergence detection
# ═════════════════════════════════════════════════════════════════════

def find_convergence_episode(
    history: List[Dict[str, Any]],
    window: int = 5,
    threshold: float = 0.05,
) -> int:
    """Find first episode where |ΔH| < threshold for *window* consecutive eps.

    Returns the episode index, or -1 if convergence not reached.
    """
    if len(history) < window + 1:
        return -1

    consecutive = 0
    for i in range(1, len(history)):
        delta_h = abs(history[i]["entropy"] - history[i - 1]["entropy"])
        if delta_h < threshold:
            consecutive += 1
            if consecutive >= window:
                return i - window + 1  # first ep of the window
        else:
            consecutive = 0

    return -1


# ═════════════════════════════════════════════════════════════════════
#  Gini coefficient
# ═════════════════════════════════════════════════════════════════════

def compute_gini(utilities: List[float]) -> float:
    """Compute the Gini coefficient for a list of utility values.

    0.0 = perfect equality, 1.0 = perfect inequality.
    Returns 0.0 for empty or all-zero inputs.
    """
    if not utilities or all(u == 0 for u in utilities):
        return 0.0
    sorted_u = sorted(utilities)
    n = len(sorted_u)
    cumsum = sum((i + 1) * u for i, u in enumerate(sorted_u))
    total = sum(sorted_u)
    if total == 0:
        return 0.0
    return (2 * cumsum) / (n * total) - (n + 1) / n


# ═════════════════════════════════════════════════════════════════════
#  Single ablation run
# ═════════════════════════════════════════════════════════════════════

@dataclass
class AblationResult:
    """Metrics collected from one ablation configuration."""
    config_name: str
    description: str
    final_rho: float = 0.0
    final_kappa: float = 0.0
    final_entropy: float = 0.0
    final_sigma_size: int = 0
    final_planning_depth: float = 0.0
    convergence_episode: int = -1
    gini_coefficient: float = 0.0
    duration_s: float = 0.0


def run_single_ablation(
    config_name: str,
    config: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    output_dir: Path,
    model_path: Optional[str] = None,
    seed: int = 42,
    episodes_per_iteration: int = 10,
    snapshot_interval: int = 10,
) -> AblationResult:
    """Run one ablation configuration and return metrics.

    Creates a fresh agent for isolation between ablation runs.
    """
    from agents.main_agent import MainAgent
    from core.config import Config

    logger.info(
        "══════════════════════════════════════════════════════════\n"
        "  Ablation: %s\n"
        "  %s\n"
        "══════════════════════════════════════════════════════════",
        config_name, config.get("description", ""),
    )

    run_dir = output_dir / config_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Build fresh agent ────────────────────────────────────────
    cfg = Config.default()
    if model_path:
        cfg.model_path = model_path

    agent = MainAgent(cfg)
    hints = configure_agent(agent, config)

    # ── Build tracker ────────────────────────────────────────────
    tracker = MetricsTracker(contraction_window=10)

    # ── Wrap agent.run to respect hints ──────────────────────────
    original_run = agent.run

    def _wrapped_run(task, **kwargs):
        if hints.get("force_strategy"):
            kwargs["strategy"] = hints["force_strategy"]
        kwargs["do_reflect"] = hints.get("do_reflect", True)
        return original_run(task, **kwargs)

    agent.run = _wrapped_run

    # ── Run experiment loop ──────────────────────────────────────
    start = time.time()
    run_experiment(
        agent=agent,
        tasks=tasks,
        tracker=tracker,
        output_dir=run_dir,
        snapshot_interval=snapshot_interval,
        episodes_per_iteration=episodes_per_iteration,
        seed=seed,
    )
    duration = time.time() - start

    # ── Extract final metrics ────────────────────────────────────
    history = tracker.get_history()
    if history:
        last = history[-1]
        final_rho = last["rho"]
        final_kappa = last["kappa"]
        final_entropy = last["entropy"]
        final_sigma_size = last["sigma_size"]
        final_depth = last["planning_depth"]
    else:
        final_rho = final_kappa = final_entropy = final_depth = 0.0
        final_sigma_size = 0

    # Convergence
    convergence_ep = find_convergence_episode(history)

    # Gini
    utilities = [s.utility for s in agent.skill_graph.skills]
    gini = compute_gini(utilities)

    result = AblationResult(
        config_name=config_name,
        description=config.get("description", ""),
        final_rho=round(final_rho, 4),
        final_kappa=round(final_kappa, 4),
        final_entropy=round(final_entropy, 4),
        final_sigma_size=final_sigma_size,
        final_planning_depth=round(final_depth, 2),
        convergence_episode=convergence_ep,
        gini_coefficient=round(gini, 4),
        duration_s=round(duration, 2),
    )

    # Save per-config result
    (run_dir / "ablation_result.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result


# ═════════════════════════════════════════════════════════════════════
#  Output: comparison CSV + text report
# ═════════════════════════════════════════════════════════════════════

def write_comparison_csv(
    results: List[AblationResult],
    filepath: str,
) -> None:
    """Write ablation_comparison.csv."""
    fields = [
        "config", "final_rho", "final_kappa", "final_entropy",
        "final_sigma_size", "final_planning_depth",
        "convergence_episode", "gini_coefficient",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "config": r.config_name,
                "final_rho": f"{r.final_rho:.3f}",
                "final_kappa": f"{r.final_kappa:.3f}",
                "final_entropy": f"{r.final_entropy:.2f}",
                "final_sigma_size": r.final_sigma_size,
                "final_planning_depth": f"{r.final_planning_depth:.1f}",
                "convergence_episode": r.convergence_episode,
                "gini_coefficient": f"{r.gini_coefficient:.2f}",
            })


def build_ablation_report(
    results: List[AblationResult],
    total_duration: float,
    seed: int,
    num_tasks: int,
) -> str:
    """Generate the ablation_report.txt text report."""
    lines = [
        "=" * 62,
        "  Self-Evolving Skill Graph — Ablation Study Report",
        "=" * 62,
        "",
        f"Total Configurations: {len(results)}",
        f"Tasks per Configuration: {num_tasks}",
        f"Total Duration: {_format_duration(total_duration)}",
        f"Seed: {seed}",
        "",
    ]

    # ── Comparison table ─────────────────────────────────────────
    lines.append("--- Comparison Table ---")
    header = (
        f"| {'Config':<25} | {'ρ':>6} | {'κ':>6} | {'H':>6} | "
        f"{'|Σ|':>4} | {'E[D]':>6} | {'Conv':>5} | {'Gini':>5} |"
    )
    lines.append(header)
    lines.append(
        f"|{'-'*27}|{'-'*8}|{'-'*8}|{'-'*8}|"
        f"{'-'*6}|{'-'*8}|{'-'*7}|{'-'*7}|"
    )
    for r in results:
        conv = str(r.convergence_episode) if r.convergence_episode >= 0 else "N/A"
        lines.append(
            f"| {r.config_name:<25} | {r.final_rho:6.3f} | "
            f"{r.final_kappa:6.3f} | {r.final_entropy:6.2f} | "
            f"{r.final_sigma_size:4d} | {r.final_planning_depth:6.1f} | "
            f"{conv:>5} | {r.gini_coefficient:5.2f} |"
        )
    lines.append("")

    # ── Impact analysis ──────────────────────────────────────────
    lines.append("--- Component Impact Analysis ---")
    full = next((r for r in results if r.config_name == "full_system"), None)

    if full:
        for r in results:
            if r.config_name == "full_system":
                continue
            lines.append("")
            lines.append(f"▸ {r.config_name}: {r.description}")

            # Compute deltas vs full
            d_rho = r.final_rho - full.final_rho
            d_kappa = r.final_kappa - full.final_kappa
            d_entropy = r.final_entropy - full.final_entropy
            d_sigma = r.final_sigma_size - full.final_sigma_size
            d_depth = r.final_planning_depth - full.final_planning_depth
            d_gini = r.gini_coefficient - full.gini_coefficient

            lines.append(f"  Δρ = {d_rho:+.3f}  Δκ = {d_kappa:+.3f}  "
                         f"ΔH = {d_entropy:+.2f}  "
                         f"Δ|Σ| = {d_sigma:+d}  "
                         f"ΔE[D] = {d_depth:+.1f}  "
                         f"ΔGini = {d_gini:+.2f}")

            # Interpret the deltas
            impacts = []
            if d_rho < -0.05:
                impacts.append("降低了 trace 壓縮效率")
            if d_kappa > 0.05:
                impacts.append("圖結構不穩定（κ 偏高）")
            if d_depth > 1.0:
                impacts.append("規劃深度增加（效率降低）")
            if d_sigma > 3:
                impacts.append("圖過大（缺乏剪枝）")
            if r.convergence_episode < 0 and full.convergence_episode >= 0:
                impacts.append("未收斂")
            if d_gini < -0.1:
                impacts.append("utility 分佈更均勻（缺乏 core-periphery 結構）")

            if impacts:
                lines.append(f"  影響：{'；'.join(impacts)}")
            else:
                lines.append("  影響：與完整系統差異不大")
    else:
        lines.append("(No full_system baseline found for comparison)")

    lines.append("")

    # ── Summary ──────────────────────────────────────────────────
    lines.append("--- Key Findings ---")
    if full:
        # Find most impactful ablation (largest rho drop)
        ablations = [r for r in results if r.config_name != "full_system"]
        if ablations:
            worst = min(ablations, key=lambda r: r.final_rho)
            lines.append(
                f"• 對 ρ 影響最大的消融：{worst.config_name} "
                f"(Δρ = {worst.final_rho - full.final_rho:+.3f})"
            )
            worst_depth = max(ablations, key=lambda r: r.final_planning_depth)
            lines.append(
                f"• 對 E[D] 影響最大的消融：{worst_depth.config_name} "
                f"(ΔE[D] = {worst_depth.final_planning_depth - full.final_planning_depth:+.1f})"
            )
            non_converged = [
                r for r in ablations if r.convergence_episode < 0
            ]
            if non_converged:
                names = ", ".join(r.config_name for r in non_converged)
                lines.append(f"• 未收斂的消融組：{names}")
            else:
                lines.append("• 所有消融組均達到收斂")

    lines.append("")
    lines.append(
        "Note: Task accuracy is NOT a formal evaluation metric. "
        "The paper evaluates based on structural criteria (ρ, κ, H, ΔΣ, E[D])."
    )

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for Skill Graph evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Run all ablation configs
  python experiments/run_ablation.py \\
      --tasks-dir experiments/tasks \\
      --output-dir experiments/results/ablation_001

  # Run specific configs only
  python experiments/run_ablation.py \\
      --tasks-dir experiments/tasks \\
      --output-dir experiments/results/ablation_002 \\
      --configs full_system no_contraction vanilla_baseline
""",
    )

    parser.add_argument(
        "--tasks-dir",
        required=True,
        help="Directory containing tier_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Specific configs to run (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--episodes-per-iteration",
        type=int,
        default=10,
        help="Episodes per reporting iteration (default: 10)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=10,
        help="Graph snapshot interval (default: 10)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to GGUF model file (overrides config)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load tasks (sequential mode, same for all configs) ───────
    curriculum = TaskCurriculum(args.tasks_dir)
    tasks = curriculum.get_sequential()
    logger.info(
        "Loaded %d tasks (sequential, seed=%d)", len(tasks), args.seed,
    )

    # ── Determine configs to run ─────────────────────────────────
    if args.configs:
        selected = {}
        for name in args.configs:
            if name not in ABLATION_CONFIGS:
                parser.error(
                    f"Unknown config: {name}. "
                    f"Available: {', '.join(ABLATION_CONFIGS.keys())}"
                )
            selected[name] = ABLATION_CONFIGS[name]
    else:
        selected = ABLATION_CONFIGS

    # ── Prepare output ───────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment metadata
    metadata = {
        "configs_run": list(selected.keys()),
        "num_tasks": len(tasks),
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "ablation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # ── Run each config ──────────────────────────────────────────
    total_start = time.time()
    results: List[AblationResult] = []

    for name, config in selected.items():
        result = run_single_ablation(
            config_name=name,
            config=config,
            tasks=tasks,
            output_dir=output_dir,
            model_path=args.model,
            seed=args.seed,
            episodes_per_iteration=args.episodes_per_iteration,
            snapshot_interval=args.snapshot_interval,
        )
        results.append(result)
        logger.info(
            "✓ %s done: ρ=%.3f κ=%.3f H=%.2f |Σ|=%d E[D]=%.1f",
            name, result.final_rho, result.final_kappa,
            result.final_entropy, result.final_sigma_size,
            result.final_planning_depth,
        )

    total_duration = time.time() - total_start

    # ── Write outputs ────────────────────────────────────────────
    write_comparison_csv(results, str(output_dir / "ablation_comparison.csv"))

    report = build_ablation_report(
        results,
        total_duration=total_duration,
        seed=args.seed,
        num_tasks=len(tasks),
    )
    (output_dir / "ablation_report.txt").write_text(
        report, encoding="utf-8",
    )
    print(report)

    logger.info("Ablation study complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
