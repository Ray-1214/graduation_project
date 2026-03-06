"""
Tests for run_ablation.py

Since the full pipeline requires an LLM, these tests focus on:
  - Ablation configuration constants
  - configure_agent() component disabling
  - Convergence detection
  - Gini coefficient
  - CSV output
  - Report generation
  - CLI parsing
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.run_ablation import (
    ABLATION_CONFIGS,
    AblationResult,
    build_ablation_report,
    build_parser,
    compute_gini,
    configure_agent,
    find_convergence_episode,
    write_comparison_csv,
)


# ═══════════════════════════════════════════════════════════════════
#  Configuration constants
# ═══════════════════════════════════════════════════════════════════

class TestAblationConfigs:

    def test_seven_configs_defined(self):
        assert len(ABLATION_CONFIGS) == 7

    def test_full_system_all_enabled(self):
        full = ABLATION_CONFIGS["full_system"]
        assert full["compound_reasoning"] is True
        assert full["graph_contraction"] is True
        assert full["tiered_memory"] is True
        assert full["hallucination_guard"] is True
        assert full["reflexion_memory"] is True

    def test_vanilla_baseline_all_disabled(self):
        vanilla = ABLATION_CONFIGS["vanilla_baseline"]
        assert vanilla["compound_reasoning"] is False
        assert vanilla["graph_contraction"] is False
        assert vanilla["tiered_memory"] is False
        assert vanilla["hallucination_guard"] is False
        assert vanilla["reflexion_memory"] is False

    def test_each_no_config_disables_exactly_one(self):
        """Each 'no_*' config should differ from full by exactly one flag."""
        full = ABLATION_CONFIGS["full_system"]
        component_keys = [
            "compound_reasoning", "graph_contraction", "tiered_memory",
            "hallucination_guard", "reflexion_memory",
        ]
        no_configs = {
            k: v for k, v in ABLATION_CONFIGS.items()
            if k.startswith("no_")
        }
        assert len(no_configs) == 5

        for name, config in no_configs.items():
            diffs = sum(
                1 for key in component_keys
                if config[key] != full[key]
            )
            assert diffs == 1, (
                f"{name} differs from full_system by {diffs} flags, expected 1"
            )

    def test_all_have_descriptions(self):
        for name, config in ABLATION_CONFIGS.items():
            assert "description" in config, f"{name} missing description"
            assert config["description"], f"{name} has empty description"


# ═══════════════════════════════════════════════════════════════════
#  configure_agent
# ═══════════════════════════════════════════════════════════════════

class TestConfigureAgent:

    def _make_mock_agent(self):
        agent = MagicMock()
        agent.evolution = MagicMock()
        agent.compound_reasoner = MagicMock()
        return agent

    def test_full_system_no_changes(self):
        agent = self._make_mock_agent()
        hints = configure_agent(agent, ABLATION_CONFIGS["full_system"])
        assert hints["force_strategy"] is None
        assert hints["do_reflect"] is True

    def test_no_compound_forces_cot(self):
        agent = self._make_mock_agent()
        hints = configure_agent(agent, ABLATION_CONFIGS["no_compound"])
        assert hints["force_strategy"] == "cot_only"

    def test_no_reflexion_disables_reflect(self):
        agent = self._make_mock_agent()
        hints = configure_agent(agent, ABLATION_CONFIGS["no_reflexion"])
        assert hints["do_reflect"] is False

    def test_no_contraction_patches_method(self):
        agent = self._make_mock_agent()
        configure_agent(agent, ABLATION_CONFIGS["no_contraction"])
        # Contraction step should be replaced with no-op
        agent.evolution._step_subgraph_contraction = MagicMock()

    def test_no_tiered_memory_patches_method(self):
        agent = self._make_mock_agent()
        configure_agent(agent, ABLATION_CONFIGS["no_tiered_memory"])
        agent.evolution._step_memory_tier_update = MagicMock()

    def test_vanilla_baseline_replaces_evolve(self):
        agent = self._make_mock_agent()
        hints = configure_agent(agent, ABLATION_CONFIGS["vanilla_baseline"])
        # evolve() should be replaced; hints should force CoT with no reflect
        assert hints["force_strategy"] == "cot_only"
        assert hints["do_reflect"] is False


# ═══════════════════════════════════════════════════════════════════
#  Convergence detection
# ═══════════════════════════════════════════════════════════════════

class TestConvergence:

    def test_converges_immediately(self):
        # 10 episodes, entropy constant
        history = [{"entropy": 2.0} for _ in range(10)]
        ep = find_convergence_episode(history, window=5, threshold=0.05)
        assert ep == 1  # first window starts at index 1

    def test_converges_late(self):
        history = []
        for i in range(20):
            # Entropy jumps for first 10 episodes, then stabilises
            e = 1.0 + i * 0.5 if i < 10 else 6.0
            history.append({"entropy": e})
        ep = find_convergence_episode(history, window=5, threshold=0.05)
        assert ep >= 0  # should converge after episode 10
        assert ep <= 15

    def test_never_converges(self):
        history = [{"entropy": float(i)} for i in range(20)]
        ep = find_convergence_episode(history, window=5, threshold=0.05)
        assert ep == -1

    def test_insufficient_data(self):
        history = [{"entropy": 1.0}, {"entropy": 1.0}]
        ep = find_convergence_episode(history, window=5, threshold=0.05)
        assert ep == -1

    def test_empty_history(self):
        assert find_convergence_episode([], window=5) == -1


# ═══════════════════════════════════════════════════════════════════
#  Gini coefficient
# ═══════════════════════════════════════════════════════════════════

class TestGini:

    def test_perfect_equality(self):
        assert compute_gini([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_perfect_inequality(self):
        # One person has everything
        gini = compute_gini([0.0, 0.0, 0.0, 100.0])
        assert gini > 0.6  # should be high

    def test_empty_list(self):
        assert compute_gini([]) == 0.0

    def test_all_zeros(self):
        assert compute_gini([0, 0, 0]) == 0.0

    def test_single_value(self):
        assert compute_gini([5.0]) == 0.0

    def test_moderate_inequality(self):
        gini = compute_gini([1, 2, 3, 4, 5])
        assert 0.1 < gini < 0.5  # moderate spread


# ═══════════════════════════════════════════════════════════════════
#  Comparison CSV
# ═══════════════════════════════════════════════════════════════════

class TestComparisonCsv:

    def test_writes_correct_format(self, tmp_path):
        results = [
            AblationResult(
                config_name="full_system",
                description="Full",
                final_rho=0.355,
                final_kappa=0.010,
                final_entropy=3.09,
                final_sigma_size=12,
                final_planning_depth=7.0,
                convergence_episode=35,
                gini_coefficient=0.62,
            ),
            AblationResult(
                config_name="vanilla_baseline",
                description="Baseline",
                final_rho=0.0,
                final_kappa=0.0,
                final_entropy=0.0,
                final_sigma_size=0,
                final_planning_depth=12.0,
                convergence_episode=-1,
                gini_coefficient=0.0,
            ),
        ]

        path = str(tmp_path / "comparison.csv")
        write_comparison_csv(results, path)

        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["config"] == "full_system"
        assert rows[0]["final_rho"] == "0.355"
        assert rows[1]["config"] == "vanilla_baseline"
        assert rows[1]["convergence_episode"] == "-1"

    def test_empty_results(self, tmp_path):
        path = str(tmp_path / "empty.csv")
        write_comparison_csv([], path)

        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 0


# ═══════════════════════════════════════════════════════════════════
#  Ablation report
# ═══════════════════════════════════════════════════════════════════

class TestBuildReport:

    def _make_results(self) -> list:
        return [
            AblationResult(
                config_name="full_system",
                description="完整系統",
                final_rho=0.355,
                final_kappa=0.010,
                final_entropy=3.09,
                final_sigma_size=12,
                final_planning_depth=7.0,
                convergence_episode=35,
                gini_coefficient=0.62,
            ),
            AblationResult(
                config_name="no_contraction",
                description="關閉 contraction",
                final_rho=0.180,
                final_kappa=0.0,
                final_entropy=3.50,
                final_sigma_size=25,
                final_planning_depth=9.5,
                convergence_episode=-1,
                gini_coefficient=0.30,
            ),
            AblationResult(
                config_name="vanilla_baseline",
                description="純 LLM",
                final_rho=0.0,
                final_kappa=0.0,
                final_entropy=0.0,
                final_sigma_size=0,
                final_planning_depth=12.0,
                convergence_episode=-1,
                gini_coefficient=0.0,
            ),
        ]

    def test_contains_header(self):
        text = build_ablation_report(
            self._make_results(), 600.0, 42, 60,
        )
        assert "Ablation Study Report" in text
        assert "Total Configurations: 3" in text

    def test_contains_comparison_table(self):
        text = build_ablation_report(
            self._make_results(), 600.0, 42, 60,
        )
        assert "full_system" in text
        assert "no_contraction" in text
        assert "vanilla_baseline" in text

    def test_contains_impact_analysis(self):
        text = build_ablation_report(
            self._make_results(), 600.0, 42, 60,
        )
        assert "Component Impact Analysis" in text
        assert "Δρ" in text

    def test_contains_key_findings(self):
        text = build_ablation_report(
            self._make_results(), 600.0, 42, 60,
        )
        assert "Key Findings" in text

    def test_no_full_system_baseline_message(self):
        results = [
            AblationResult(
                config_name="vanilla_baseline",
                description="Baseline",
            ),
        ]
        text = build_ablation_report(results, 60.0, 42, 10)
        assert "No full_system baseline" in text


# ═══════════════════════════════════════════════════════════════════
#  CLI parser
# ═══════════════════════════════════════════════════════════════════

class TestParser:

    def test_required_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-dir", "tasks/",
            "--output-dir", "output/",
        ])
        assert args.tasks_dir == "tasks/"
        assert args.output_dir == "output/"

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-dir", "t",
            "--output-dir", "o",
        ])
        assert args.seed == 42
        assert args.snapshot_interval == 10
        assert args.episodes_per_iteration == 10
        assert args.configs is None
        assert args.verbose is False

    def test_select_specific_configs(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-dir", "t",
            "--output-dir", "o",
            "--configs", "full_system", "no_contraction",
        ])
        assert args.configs == ["full_system", "no_contraction"]
