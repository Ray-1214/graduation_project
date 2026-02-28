"""
Tree of Thoughts — multi-branch search reasoning.

Implements BFS and DFS search over a tree of reasoning steps.
At each node, the LLM generates N candidate next-steps (branches),
an evaluator scores them, and the search prunes/selects the best
paths to continue exploring.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from core.llm_interface import BaseLLM
from core.prompt_builder import PromptBuilder
from memory.episodic_log import EpisodicLog

logger = logging.getLogger(__name__)


@dataclass
class ThoughtNode:
    """A node in the thought tree."""
    thought: str
    score: float = 0.0
    depth: int = 0
    children: List["ThoughtNode"] = field(default_factory=list)
    parent: Optional["ThoughtNode"] = None

    def path_str(self) -> str:
        """Reconstruct the reasoning path from root to this node."""
        nodes = []
        node = self
        while node is not None:
            nodes.append(node.thought)
            node = node.parent
        nodes.reverse()
        return "\n→ ".join(nodes)


class TreeOfThoughts:
    """Multi-branch search over reasoning paths."""

    def __init__(
        self,
        llm: BaseLLM,
        prompt_builder: PromptBuilder,
        branch_factor: int = 3,
        max_depth: int = 3,
        beam_width: int = 2,
    ) -> None:
        self.llm = llm
        self.pb = prompt_builder
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.beam_width = beam_width

    # -- core operations --

    def expand(self, task: str, node: ThoughtNode) -> List[str]:
        """Generate candidate next thoughts from a node."""
        prompt = self.pb.build(
            "tot_expand",
            task=task,
            current_path=node.path_str(),
            n_branches=self.branch_factor,
        )
        response = self.llm.generate(prompt, max_tokens=512)

        # Parse "Step N: ..." lines
        branches = re.findall(r"Step \d+:\s*(.+)", response)
        if not branches:
            # Fallback: split on newlines
            branches = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip()
            ]
        return branches[: self.branch_factor]

    def evaluate(self, task: str, node: ThoughtNode) -> float:
        """Score a reasoning path using LLM-as-judge."""
        prompt = self.pb.build(
            "tot_evaluate",
            task=task,
            reasoning_path=node.path_str(),
        )
        response = self.llm.generate(prompt, max_tokens=16, temperature=0.1)

        # Extract numeric score
        match = re.search(r"(\d+(?:\.\d+)?)", response)
        if match:
            score = float(match.group(1))
            return min(score / 10.0, 1.0)  # normalize to 0-1
        return 0.5  # default if parsing fails

    # -- search strategies --

    def search(
        self,
        task: str,
        strategy: Literal["bfs", "dfs"] = "bfs",
        episode: Optional[EpisodicLog] = None,
    ) -> str:
        """Run tree search and return the best reasoning path."""
        root = ThoughtNode(thought=f"Problem: {task}", depth=0)

        if episode:
            episode.log_step("thought", f"ToT ({strategy}): starting search")

        if strategy == "bfs":
            best = self._bfs(task, root, episode)
        else:
            best = self._dfs(task, root, 0, episode)

        result = best.path_str() if best else "No solution found."

        if episode:
            episode.log_step("finish", result, score=best.score if best else 0.0)

        return result

    def _bfs(
        self,
        task: str,
        root: ThoughtNode,
        episode: Optional[EpisodicLog] = None,
    ) -> ThoughtNode:
        """Beam search (BFS with pruning)."""
        current_level = [root]
        best_node = root

        for depth in range(self.max_depth):
            candidates: List[ThoughtNode] = []

            for node in current_level:
                branches = self.expand(task, node)
                for thought in branches:
                    child = ThoughtNode(
                        thought=thought,
                        depth=depth + 1,
                        parent=node,
                    )
                    child.score = self.evaluate(task, child)
                    node.children.append(child)
                    candidates.append(child)

                    if episode:
                        episode.log_step(
                            "branch",
                            f"[d={depth + 1}] {thought} (score={child.score:.2f})",
                        )

            if not candidates:
                break

            # Prune: keep top beam_width nodes
            candidates.sort(key=lambda n: n.score, reverse=True)
            current_level = candidates[: self.beam_width]

            if current_level[0].score > best_node.score:
                best_node = current_level[0]

            logger.info(
                "ToT BFS depth %d: %d candidates, best=%.2f",
                depth + 1, len(candidates), best_node.score,
            )

        return best_node

    def _dfs(
        self,
        task: str,
        node: ThoughtNode,
        depth: int,
        episode: Optional[EpisodicLog] = None,
    ) -> ThoughtNode:
        """Depth-first search with scoring."""
        if depth >= self.max_depth:
            node.score = self.evaluate(task, node)
            return node

        branches = self.expand(task, node)
        best_child = node

        for thought in branches:
            child = ThoughtNode(
                thought=thought,
                depth=depth + 1,
                parent=node,
            )
            node.children.append(child)

            if episode:
                episode.log_step("branch", f"[d={depth + 1}] {thought}")

            result = self._dfs(task, child, depth + 1, episode)
            if result.score > best_child.score:
                best_child = result

        return best_child
