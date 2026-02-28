"""
Experiment runner — CLI entry point for running reasoning experiments.

Usage:
    python experiments/run_experiment.py \\
        --task "What is 25 * 17?" \\
        --strategy react \\
        --use-rag \\
        --no-reflect
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Config
from agents.main_agent import MainAgent


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def save_result(result, results_dir: str) -> str:
    """Save experiment result to a timestamped JSON file."""
    p = Path(results_dir)
    p.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time())}_{result.strategy}.json"
    filepath = p / filename
    filepath.write_text(json.dumps(asdict(result), indent=2, default=str))
    return str(filepath)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reasoning experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chain of Thought
  python experiments/run_experiment.py --task "Explain quantum entanglement" --strategy cot

  # ReAct with tool use
  python experiments/run_experiment.py --task "What is 25 * 17?" --strategy react

  # Tree of Thoughts
  python experiments/run_experiment.py --task "List 3 ways to reduce energy usage" --strategy tot

  # Auto-select strategy with RAG
  python experiments/run_experiment.py --task "Summarize the key findings" --use-rag
        """,
    )

    parser.add_argument(
        "--task", "-t",
        required=True,
        help="The task/question to reason about",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["cot", "tot", "react", "auto"],
        default="auto",
        help="Reasoning strategy (default: auto-select)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to GGUF model file (overrides config)",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Enable RAG context injection",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory to index for RAG before running",
    )
    parser.add_argument(
        "--no-reflect",
        action="store_true",
        help="Skip reflexion after the task",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Build config
    config = Config.default()
    if args.model:
        config.model_path = args.model
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    if args.temperature is not None:
        config.temperature = args.temperature

    # Init agent
    logger = logging.getLogger("experiment")
    logger.info("Initializing MainAgent …")
    agent = MainAgent(config)

    # Index documents if requested
    if args.index_dir:
        logger.info("Indexing documents from %s …", args.index_dir)
        count = agent.index_knowledge(args.index_dir)
        logger.info("Indexed %d chunks.", count)
    elif args.use_rag:
        # Auto-index the default knowledge base
        kb_dir = config.knowledge_base_dir
        if Path(kb_dir).is_dir() and any(Path(kb_dir).iterdir()):
            logger.info("Auto-indexing knowledge base at %s …", kb_dir)
            count = agent.index_knowledge(kb_dir)
            logger.info("Indexed %d chunks.", count)

    # Resolve strategy
    strategy = None if args.strategy == "auto" else args.strategy

    # Run
    logger.info("=" * 60)
    logger.info("Task: %s", args.task)
    logger.info("Strategy: %s", strategy or "auto-select")
    logger.info("=" * 60)

    result = agent.run(
        task=args.task,
        strategy=strategy,
        use_rag=args.use_rag,
        do_reflect=not args.no_reflect,
    )

    # Save results
    filepath = save_result(result, config.results_dir)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Strategy : {result.strategy}")
    print(f"Score    : {result.score:.2f}" if result.score else "Score    : N/A")
    print(f"Duration : {result.duration_s}s")
    print(f"Saved to : {filepath}")
    print("=" * 60)
    print(f"\nAnswer:\n{result.answer}")

    if result.reflection:
        print(f"\nReflection:\n{result.reflection}")


if __name__ == "__main__":
    main()
