"""
Main Agent — top-level orchestrator for the cognitive architecture.

Routes tasks to reasoning strategies, optionally injects RAG context,
runs evaluation and reflexion, and returns structured results.

Phase 2 additions:
  - SkillGraph integration: retrieves relevant learned skills before
    reasoning and evolves the graph after each episode.

Phase 3.5 additions:
  - CompoundReasoner replaces _dispatch() — unified multi-phase engine.
  - HallucinationGuard — 3-stage fact grounding pipeline.
  - ContextAssembler — slot-based context window management.
  - ReflexionMemoryWriter — routes reflexion insights to memory stores.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.config import Config
from core.llm_interface import BaseLLM, LlamaCppLLM
from core.prompt_builder import PromptBuilder
from memory.episodic_log import EpisodicLog, convert_log_to_trace
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from memory.vector_store import VectorStore
from memory.reflexion_memory_writer import ReflexionMemoryWriter
from rag.indexer import Indexer
from rag.retriever import Retriever
from reasoning.cot import ChainOfThought
from reasoning.context_assembler import ContextAssembler
from reasoning.compound_reasoner import CompoundReasoner
from reasoning.hallucination_guard import HallucinationGuard
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
from skill_graph.memory_partition import MemoryPartition
from skill_graph.evolution_operator import EvolutionOperator
from skill_graph.skill_retriever import SkillRetriever

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
    # Phase 3.5: CompoundReasoner additions
    verification_verdict: Optional[str] = None
    hallucination_score: Optional[float] = None


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
        self.skill_retriever = SkillRetriever(top_k=3)

        # ── Phase 3.5: Compound Reasoning Pipeline ───────────────
        self.hallucination_guard = HallucinationGuard(
            llm=self.llm,
            knowledge_store=self.knowledge_store,
        )
        self.context_assembler = ContextAssembler(
            max_total_tokens=4000,
            llm=self.llm,
        )
        self.compound_reasoner = CompoundReasoner(
            llm=self.llm,
            prompt_builder=self.pb,
            skill_registry=self.skills,
            cot=self.cot,
            tot=self.tot,
            react=self.react,
            reflexion=self.reflexion,
            hallucination_guard=self.hallucination_guard,
            context_assembler=self.context_assembler,
            knowledge_store=self.knowledge_store,
        )
        self.reflexion_writer = ReflexionMemoryWriter(
            long_term=self.long_term,
            knowledge_store=self.knowledge_store,
        )

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
                      Auto-selected via Phase ① if None.
            use_rag: Whether to inject RAG context before reasoning.
            do_reflect: Whether to run Reflexion after the task.

        Returns:
            AgentResult with answer, score, reflection, and trace.
        """
        start_time = time.time()

        # 1. Select strategy (for logging; CompoundReasoner may override)
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
        retrieved_skills = []
        if len(self.skill_graph) > 0:
            retrieved_skills = self.skill_retriever.retrieve(
                task, self.skill_graph,
            )
            if retrieved_skills:
                skill_block = self.skill_retriever.format_for_prompt(
                    retrieved_skills,
                )
                effective_task = f"{skill_block}\n\n{effective_task}"
                episode.log_step(
                    "thought",
                    f"Injected {len(retrieved_skills)} learned skill(s) from graph",
                )

        # ── Phase 3.5: CompoundReasoner replaces _dispatch() ─────
        # force_strategy=None  → Phase ① auto-selects
        # force_strategy="cot" → backward-compatible forced mode
        result = self.compound_reasoner.run(
            effective_task, episode,
            force_strategy=strategy,  # None = auto, else forced
        )
        answer = result.answer

        # Record verification metadata to episode
        verification_verdict = None
        hallucination_score = None
        if result.verification:
            verification_verdict = result.verification.verdict
            hallucination_score = result.verification.hallucination_score
            episode.log_step(
                "verification",
                f"verdict={verification_verdict}, "
                f"hallucination_score={hallucination_score}",
            )

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

            # Phase 3.5: Route reflexion insights to memory systems
            try:
                commit = self.reflexion_writer.process(
                    reflexion_text=reflection_text,
                    trace=trace,
                    success=trace.success,
                    used_skills=retrieved_skills or None,
                )
                logger.info(
                    "ReflexionMemoryWriter: lessons=%d, knowledge=%d, "
                    "warnings=%d",
                    commit.strategy_lessons_written,
                    commit.knowledge_gains_written,
                    commit.warnings_dispatched,
                )
            except Exception as exc:
                logger.warning("ReflexionMemoryWriter failed: %s", exc)

        duration = time.time() - start_time

        return AgentResult(
            task=task,
            strategy=result.strategy_used,
            answer=answer,
            evolution_log=evo_log.summary(),
            score=score,
            reflection=reflection_text,
            episode=episode.to_dict(),
            duration_s=round(duration, 2),
            verification_verdict=verification_verdict,
            hallucination_score=hallucination_score,
        )

    def index_knowledge(self, path: str) -> int:
        """Index documents for RAG. Accepts file or directory paths."""
        from pathlib import Path
        p = Path(path)
        if p.is_dir():
            return self.indexer.index_directory(path)
        return self.indexer.index_file(path)

    # ── internals (deprecated — kept for reference) ──────────────────

    def _dispatch(
        self,
        strategy: str,
        task: str,
        episode: EpisodicLog,
    ) -> str:
        """Dispatch to the appropriate reasoning strategy.

        .. deprecated::
            Replaced by :meth:`CompoundReasoner.run`. Kept for
            backward compatibility if called directly.
        """
        if strategy == "cot":
            return self.cot.run(task, episode)
        elif strategy == "tot":
            return self.tot.search(task, strategy="bfs", episode=episode)
        elif strategy == "react":
            return self.react.run(task, episode)
        else:
            logger.warning("Unknown strategy '%s', falling back to CoT.", strategy)
            return self.cot.run(task, episode)
