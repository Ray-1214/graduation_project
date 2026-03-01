"""
Main Agent — top-level orchestrator for the cognitive architecture.

Routes tasks to reasoning strategies, optionally injects RAG context,
runs evaluation and reflexion, and returns structured results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.config import Config
from core.llm_interface import BaseLLM, LlamaCppLLM
from core.prompt_builder import PromptBuilder
from memory.episodic_log import EpisodicLog
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

        # 5. Run the selected strategy
        answer = self._dispatch(strat, effective_task, episode)

        # 6. Evaluate
        score = self.evaluator.evaluate(task, answer)
        episode.finish(result=answer, score=score)

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
