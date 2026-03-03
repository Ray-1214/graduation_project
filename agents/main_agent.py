"""
Main Agent — top-level orchestrator for the cognitive architecture.

Routes tasks to reasoning strategies, optionally injects RAG context,
runs evaluation and reflexion, and returns structured results.

Phase 2 additions:
  - SkillGraph integration: retrieves relevant learned skills before
    reasoning and evolves the graph after each episode.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx  # type: ignore

from core.config import Config
from core.llm_interface import BaseLLM, LlamaCppLLM
from core.prompt_builder import PromptBuilder
from memory.episodic_log import EpisodicLog, convert_log_to_trace
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from memory.vector_store import VectorStore
from rag.indexer import Indexer
from rag.retriever import Retriever
from reasoning.cot import ChainOfThought
from reasoning.planner import StrategyPlanner
from reasoning.react import ReActLoop
from reasoning.reflexion import Reflexion
from reasoning.tot import TreeOfThoughts
from agents.evaluator_agent import EvaluatorAgent
from skills.calculator import Calculator
from skills.file_ops import FileOps
from skills.registry import SkillRegistry
from skills.web_search import WebSearch
from skills.admin_query import AdminQuery
from rag.knowledge_store import KnowledgeStore
from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode
from skill_graph.memory_partition import MemoryPartition
from skill_graph.evolution_operator import EvolutionOperator

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Structured result from a reasoning episode."""
    task: str
    strategy: str
    answer: str
    score: Optional[float] = None
    reflection: Optional[str] = None
    episode: Optional[Dict[str, Any]] = None
    duration_s: Optional[float] = None
    evolution_log: Optional[str] = None


class MainAgent:
    """Top-level orchestrator — wires all components together."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.default()

        # Core
        self.llm: BaseLLM = LlamaCppLLM(self.config)
        self.pb = PromptBuilder()

        # Memory
        self.short_term = ShortTermMemory(capacity=self.config.short_term_capacity)
        self.long_term = LongTermMemory(store_path=self.config.long_term_store_path)

        # RAG
        self.vector_store = VectorStore(
            embedding_model_name=self.config.embedding_model_name,
        )
        self.indexer = Indexer(
            self.vector_store,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.retriever = Retriever(
            self.vector_store,
            top_k=self.config.retrieval_top_k,
        )

        # Knowledge
        self.knowledge_store = KnowledgeStore(use_vectors=False)

        # Skills
        self.skills = SkillRegistry()
        self.skills.register(Calculator())
        self.skills.register(FileOps())
        self.skills.register(WebSearch(knowledge_store=self.knowledge_store))
        self.skills.register(AdminQuery(knowledge_store=self.knowledge_store))

        # Reasoning strategies
        self.planner = StrategyPlanner()
        self.cot = ChainOfThought(self.llm, self.pb)
        self.tot = TreeOfThoughts(
            self.llm, self.pb,
            branch_factor=self.config.tot_branch_factor,
            max_depth=self.config.tot_max_depth,
            beam_width=self.config.tot_beam_width,
        )
        self.react = ReActLoop(
            self.llm, self.pb, self.skills,
            max_steps=self.config.react_max_steps,
        )
        self.reflexion = Reflexion(self.llm, self.pb, self.long_term)

        # Evaluator
        self.evaluator = EvaluatorAgent(self.llm, self.pb)

        # Skill Graph + Evolution (Phase 2)
        self.skill_graph = SkillGraph()
        self.memory_partition = MemoryPartition(
            theta_high=0.7, theta_low=0.3,
            epsilon_h=0.05, epsilon_l=0.05,
        )
        self.evolution = EvolutionOperator()

    # ── public API ───────────────────────────────────────────────────

    def run(
        self,
        task: str,
        strategy: Optional[str] = None,
        use_rag: bool = False,
        do_reflect: bool = True,
    ) -> AgentResult:
        """Execute a reasoning task end-to-end.

        Args:
            task: The problem or question to reason about.
            strategy: Force a specific strategy (cot/tot/react).
                      Auto-selected if None.
            use_rag: Whether to inject RAG context before reasoning.
            do_reflect: Whether to run Reflexion after the task.

        Returns:
            AgentResult with answer, score, reflection, and trace.
        """
        start_time = time.time()

        # 1. Select strategy
        strat = strategy or self.planner.select_strategy(task)
        logger.info("Running task with strategy: %s", strat)

        # 2. Create episode log
        episode = EpisodicLog(task=task, strategy=strat)

        # 3. Optionally inject RAG context
        effective_task = task
        if use_rag:
            context = self.retriever.retrieve_context(task)
            effective_task = self.pb.build(
                "rag_context",
                task=task,
                retrieved_context=context,
            )
            episode.log_step("thought", f"RAG context injected ({len(context)} chars)")

        # 4. Inject past reflexion lessons
        lessons = self.reflexion.get_relevant_lessons(task)
        if lessons:
            effective_task = f"{lessons}\n\n{effective_task}"
            episode.log_step("thought", "Past reflexion lessons injected")

        # 4.5  Skill retrieval from graph (Phase 2)
        if len(self.skill_graph) > 0:
            retrieved = self._retrieve_skills(task)
            if retrieved:
                skill_block = self._format_skills_for_prompt(retrieved)
                effective_task = f"{skill_block}\n\n{effective_task}"
                episode.log_step(
                    "thought",
                    f"Injected {len(retrieved)} learned skill(s) from graph",
                )

        # 5. Run the selected strategy
        answer = self._dispatch(strat, effective_task, episode)

        # 6. Evaluate
        score = self.evaluator.evaluate(task, answer)
        episode.finish(result=answer, score=score)

        # 6.5  Evolution step (Phase 2)
        trace = convert_log_to_trace(
            episode,
            task_id=f"ep-{uuid.uuid4().hex[:8]}",
        )
        evo_log = self.evolution.evolve(
            self.skill_graph, trace, self.memory_partition,
        )
        logger.info("Evolution: %s", evo_log.summary())

        # 7. Reflexion (optional)
        reflection_text = None
        if do_reflect:
            entry = self.reflexion.reflect(task, episode, answer, score)
            reflection_text = entry.reflection

        duration = time.time() - start_time

        return AgentResult(
            task=task,
            strategy=strat,
            answer=answer,
            evolution_log=evo_log.summary(),
            score=score,
            reflection=reflection_text,
            episode=episode.to_dict(),
            duration_s=round(duration, 2),
        )

    def index_knowledge(self, path: str) -> int:
        """Index documents for RAG. Accepts file or directory paths."""
        from pathlib import Path
        p = Path(path)
        if p.is_dir():
            return self.indexer.index_directory(path)
        return self.indexer.index_file(path)

    # ── internals ────────────────────────────────────────────────────

    def _dispatch(
        self,
        strategy: str,
        task: str,
        episode: EpisodicLog,
    ) -> str:
        """Dispatch to the appropriate reasoning strategy."""
        if strategy == "cot":
            return self.cot.run(task, episode)
        elif strategy == "tot":
            return self.tot.search(task, strategy="bfs", episode=episode)
        elif strategy == "react":
            return self.react.run(task, episode)
        else:
            logger.warning("Unknown strategy '%s', falling back to CoT.", strategy)
            return self.cot.run(task, episode)

    # ── Skill Graph helpers (Phase 2) ────────────────────────────────

    def _retrieve_skills(
        self,
        task: str,
        top_k: int = 3,
        lambda1: float = 0.5,
        lambda2: float = 0.3,
        lambda3: float = 0.2,
    ) -> List[Tuple[SkillNode, float]]:
        """Retrieve top-k skills by activation score.

        activation(σ) = λ₁·sim(task, σ) + λ₂·U(σ) + λ₃·centrality(σ)

        Args:
            task:    Task description string.
            top_k:   Number of skills to retrieve.
            lambda1: Weight for textual similarity.
            lambda2: Weight for normalised utility.
            lambda3: Weight for PageRank centrality.

        Returns:
            List of (SkillNode, activation_score) tuples, sorted
            descending by activation score.
        """
        if len(self.skill_graph) == 0:
            return []

        # Pre-compute centrality once
        try:
            centrality = nx.pagerank(self.skill_graph._graph, alpha=0.85)
        except nx.NetworkXError:
            centrality = {nid: 0.0 for nid in self.skill_graph._graph.nodes}

        # Normalise utilities across graph
        utilities = [s.utility for s in self.skill_graph.skills]
        max_u = max(utilities) if utilities else 1.0
        max_u = max(max_u, 1e-9)  # avoid division by zero

        scored: List[Tuple[SkillNode, float]] = []
        task_lower = task.lower()

        for skill in self.skill_graph.skills:
            # Textual similarity: keyword overlap via SequenceMatcher
            sim = SequenceMatcher(
                None, task_lower,
                " ".join(skill.initiation_set + [skill.name]).lower(),
            ).ratio()

            # Normalised utility
            norm_u = skill.utility / max_u

            # Centrality
            cent = centrality.get(skill.skill_id, 0.0)

            activation = lambda1 * sim + lambda2 * norm_u + lambda3 * cent
            scored.append((skill, activation))

        # Sort descending, take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _format_skills_for_prompt(
        skills_with_scores: List[Tuple[SkillNode, float]],
    ) -> str:
        """Format retrieved skills as a prompt block."""
        lines = ["[已學到的策略 — 你可以直接使用以下技能]"]
        for i, (skill, score) in enumerate(skills_with_scores, 1):
            lines.append(
                f"\n策略 {i}: {skill.name} "
                f"(activation={score:.2f}, utility={skill.utility:.2f})"
            )
            lines.append(f"  適用: {', '.join(skill.initiation_set)}")
            lines.append(f"  步驟: {skill.policy}")
            lines.append(f"  終止: {skill.termination}")
        return "\n".join(lines)

