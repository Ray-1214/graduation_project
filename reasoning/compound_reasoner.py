"""
CompoundReasoner — unified multi-phase reasoning engine.

Replaces ``MainAgent._dispatch()`` with a compound pipeline::

    Phase ① Structured Thinking  → task analysis + plan
    Phase ② Adaptive Execution   → CoT / ReAct+CoT / ReAct+ToT / full compound
    Phase ③ Answer Verification   → hallucination guard
    Phase ④ Reflexion             → (handled externally by MainAgent)

Key features:
  - **Think Step** (Anthropic "think tool" pattern): after each
    observation, an LLM analysis step runs without producing an
    action, logged as ``type="think"``.
  - **Grounded Self-Check**: every ``check_interval`` steps,
    cross-validates reasoning against KnowledgeStore and tool
    outputs, per Huang et al. (ICLR 2024, arXiv:2310.01798).
  - **Backward compatibility**: ``force_strategy`` skips Phase ① and
    delegates to the single-strategy path.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List, Optional

from reasoning.compound_result import (
    CompoundResult,
    SelfCheckResult,
    ThinkingResult,
    VerificationResult,
    VERDICT_PASS,
)

if TYPE_CHECKING:
    from core.llm_interface import BaseLLM
    from core.prompt_builder import PromptBuilder
    from memory.episodic_log import EpisodicLog
    from rag.knowledge_store import KnowledgeStore
    from reasoning.cot import ChainOfThought
    from reasoning.context_assembler import ContextAssembler
    from reasoning.hallucination_guard import HallucinationGuard
    from reasoning.react import ReActLoop
    from reasoning.reflexion import Reflexion
    from reasoning.tot import TreeOfThoughts
    from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# Strategy constants
STRATEGY_COT = "cot_only"
STRATEGY_REACT_COT = "react_cot"
STRATEGY_REACT_TOT = "react_tot"
STRATEGY_FULL = "full_compound"

_VALID_STRATEGIES = {STRATEGY_COT, STRATEGY_REACT_COT,
                     STRATEGY_REACT_TOT, STRATEGY_FULL}

# Execution limits
_DEFAULT_MAX_STEPS = 10


class CompoundReasoner:
    """Unified multi-phase reasoning engine.

    Args:
        llm:                  LLM backend.
        prompt_builder:       Template-based prompt builder.
        skill_registry:       Tool registry for ReAct loops.
        cot:                  Chain of Thought reasoner.
        tot:                  Tree of Thoughts reasoner.
        react:                ReAct loop reasoner.
        reflexion:            Post-task reflection engine.
        hallucination_guard:  Phase ③ answer verifier.
        context_assembler:    Prompt context combiner.
        knowledge_store:      Optional knowledge store for self-check.
        check_interval:       Steps between grounded self-checks.
        max_steps:            Maximum execution steps.
    """

    def __init__(
        self,
        llm: "BaseLLM",
        prompt_builder: "PromptBuilder",
        skill_registry: "SkillRegistry",
        cot: "ChainOfThought",
        tot: "TreeOfThoughts",
        react: "ReActLoop",
        reflexion: "Reflexion",
        hallucination_guard: "HallucinationGuard",
        context_assembler: "ContextAssembler",
        knowledge_store: Optional["KnowledgeStore"] = None,
        check_interval: int = 3,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        self.llm = llm
        self.pb = prompt_builder
        self.skills = skill_registry
        self.cot = cot
        self.tot = tot
        self.react = react
        self.reflexion = reflexion
        self.guard = hallucination_guard
        self.assembler = context_assembler
        self.knowledge_store = knowledge_store
        self.check_interval = max(1, check_interval)
        self.max_steps = max_steps

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def run(
        self,
        task: str,
        episode: "EpisodicLog",
        force_strategy: Optional[str] = None,
    ) -> CompoundResult:
        """Unified reasoning entry point.

        Args:
            task:            The problem or question.
            episode:         Episodic log to record all steps.
            force_strategy:  If set, skip Phase ① and use this single
                             strategy directly (backward-compatible).

        Returns:
            :class:`CompoundResult` with answer, confidence,
            and verification report.
        """
        # ── Backward-compatible forced strategy ──────────────────
        if force_strategy:
            return self._run_forced(task, episode, force_strategy)

        # ── Phase ① Structured Thinking ─────────────────────────
        thinking = self._structured_thinking(task, episode)

        # ── Phase ② Adaptive Execution ──────────────────────────
        answer, steps_taken, think_count, check_count = (
            self._adaptive_execute(task, thinking, episode)
        )

        # ── Phase ③ Answer Verification ─────────────────────────
        # Simple tasks skip verification to save tokens.
        if thinking.complexity == "simple":
            return CompoundResult(
                answer=answer,
                confidence=1.0,
                strategy_used=thinking.recommended_strategy,
                thinking=thinking,
                verification=None,
                steps_taken=steps_taken,
                think_steps=think_count,
                self_checks=check_count,
            )

        trace_observations = self._collect_observations(episode)
        verification = self._verify_answer(
            answer, task, trace_observations, episode,
        )

        # Apply verification: tag flagged claims
        final_answer = self._apply_flags(answer, verification)

        return CompoundResult(
            answer=final_answer,
            confidence=verification.confidence,
            strategy_used=thinking.recommended_strategy,
            thinking=thinking,
            verification=verification,
            steps_taken=steps_taken,
            think_steps=think_count,
            self_checks=check_count,
        )

    # ═══════════════════════════════════════════════════════════════
    #  Phase ① — Structured Thinking
    # ═══════════════════════════════════════════════════════════════

    def _structured_thinking(
        self,
        task: str,
        episode: "EpisodicLog",
    ) -> ThinkingResult:
        """Analyse the task and produce an execution plan.

        Prompts the LLM to output a structured analysis with:
        task type, complexity, tools, strategy, and sub-goals.
        """
        prompt = (
            "分析以下任務並回答（必須嚴格遵守格式）：\n\n"
            f"任務：{task}\n\n"
            "1. 任務類型：(事實查詢 / 推理計算 / 多步驟操作 / 創意生成)\n"
            "2. 預估複雜度：(simple / moderate / complex)\n"
            "3. 需要的工具：(web_search / calculator / file_ops / "
            "admin_query / none)（可多選，用逗號分隔）\n"
            "4. 推薦策略路線：(cot_only / react_cot / react_tot / "
            "full_compound)\n"
            "5. 子目標拆解：[step1, step2, ...]\n"
        )

        try:
            raw = self.llm.generate(prompt, max_tokens=512)
        except Exception as exc:
            logger.warning("Structured thinking failed: %s", exc)
            raw = ""

        result = self._parse_thinking(raw)

        episode.log_step("thinking_plan", result.raw_analysis)
        logger.info(
            "Phase ①: type=%s complexity=%s strategy=%s sub_goals=%d",
            result.task_type, result.complexity,
            result.recommended_strategy, len(result.sub_goals),
        )

        return result

    def _parse_thinking(self, raw: str) -> ThinkingResult:
        """Parse the structured thinking LLM output."""
        result = ThinkingResult(raw_analysis=raw)

        if not raw.strip():
            return result

        # Task type
        m = re.search(r"任務類型[：:]\s*(.+)", raw)
        if m:
            result.task_type = m.group(1).strip().strip("()")

        # Complexity
        m = re.search(r"預估複雜度[：:]\s*\(?(\w+)\)?", raw)
        if m:
            val = m.group(1).lower()
            if val in ("simple", "moderate", "complex"):
                result.complexity = val

        # Tools
        m = re.search(r"需要的工具[：:]\s*(.+)", raw)
        if m:
            tools_str = m.group(1).strip().strip("()")
            if tools_str.lower() not in ("none", "無"):
                result.tools_needed = [
                    t.strip() for t in tools_str.split(",") if t.strip()
                ]

        # Strategy
        m = re.search(r"推薦策略路線[：:]\s*\(?(\w+)\)?", raw)
        if m:
            val = m.group(1).strip()
            if val in _VALID_STRATEGIES:
                result.recommended_strategy = val

        # Sub-goals
        m = re.search(r"子目標拆解[：:]\s*\[(.+?)\]", raw, re.DOTALL)
        if m:
            goals = [g.strip().strip("\"'") for g in m.group(1).split(",")]
            result.sub_goals = [g for g in goals if g]

        return result

    # ═══════════════════════════════════════════════════════════════
    #  Phase ② — Adaptive Execution
    # ═══════════════════════════════════════════════════════════════

    def _adaptive_execute(
        self,
        task: str,
        plan: ThinkingResult,
        episode: "EpisodicLog",
    ) -> tuple[str, int, int, int]:
        """Execute the task using the strategy chosen by Phase ①.

        Returns:
            (answer, steps_taken, think_count, check_count)
        """
        strategy = plan.recommended_strategy
        episode.log_step(
            "strategy_switch",
            f"Selected strategy: {strategy}",
        )

        if strategy == STRATEGY_COT:
            answer = self._exec_cot(task, episode)
            return answer, 1, 0, 0

        elif strategy == STRATEGY_REACT_COT:
            return self._exec_react_cot(task, plan, episode)

        elif strategy == STRATEGY_REACT_TOT:
            return self._exec_react_tot(task, plan, episode)

        elif strategy == STRATEGY_FULL:
            return self._exec_full_compound(task, plan, episode)

        else:
            # Fallback
            logger.warning("Unknown strategy '%s', falling back to CoT.", strategy)
            answer = self._exec_cot(task, episode)
            return answer, 1, 0, 0

    # ── cot_only ─────────────────────────────────────────────────

    def _exec_cot(self, task: str, episode: "EpisodicLog") -> str:
        """Simple single-pass CoT."""
        return self.cot.run(task, episode)

    # ── react_cot ────────────────────────────────────────────────

    def _exec_react_cot(
        self,
        task: str,
        plan: ThinkingResult,
        episode: "EpisodicLog",
    ) -> tuple[str, int, int, int]:
        """ReAct loop where each Thought uses CoT."""
        return self._react_loop(task, plan, episode, use_tot=False)

    # ── react_tot ────────────────────────────────────────────────

    def _exec_react_tot(
        self,
        task: str,
        plan: ThinkingResult,
        episode: "EpisodicLog",
    ) -> tuple[str, int, int, int]:
        """ReAct loop; branch points use ToT beam search."""
        return self._react_loop(task, plan, episode, use_tot=True)

    # ── full_compound ────────────────────────────────────────────

    def _exec_full_compound(
        self,
        task: str,
        plan: ThinkingResult,
        episode: "EpisodicLog",
    ) -> tuple[str, int, int, int]:
        """Full compound: ReAct + CoT + ToT + self-check."""
        return self._react_loop(
            task, plan, episode, use_tot=True, enable_self_check=True,
        )

    # ── Shared ReAct loop core ───────────────────────────────────

    def _react_loop(
        self,
        task: str,
        plan: ThinkingResult,
        episode: "EpisodicLog",
        *,
        use_tot: bool = False,
        enable_self_check: bool = False,
    ) -> tuple[str, int, int, int]:
        """Core ReAct loop with optional Think Steps and Self-Check.

        Returns:
            (answer, steps_taken, think_count, check_count)
        """
        previous_steps: str = ""
        steps_record: List[str] = []
        tool_descriptions = self.skills.list_descriptions()
        think_count = 0
        check_count = 0
        step_num = 0

        for step_num in range(1, self.max_steps + 1):
            logger.info("CompoundReasoner step %d/%d", step_num, self.max_steps)

            # ── Grounded Self-Check ──────────────────────────────
            if (enable_self_check
                    and step_num > 1
                    and (step_num - 1) % self.check_interval == 0):
                sc_result = self._grounded_self_check(
                    task, steps_record, plan, episode,
                )
                check_count += 1
                if sc_result.needs_replan:
                    # Dynamic replanning: append issues to prompt
                    replan_note = (
                        "[Self-Check 發現問題]\n"
                        + "\n".join(f"- {i}" for i in sc_result.issues)
                        + "\n請根據以上問題調整策略。\n"
                    )
                    previous_steps += replan_note
                    episode.log_step("replan", replan_note.strip())

            # ── Generate Thought ─────────────────────────────────
            thought_prompt = self.pb.build(
                "react",
                task=task,
                tool_descriptions=tool_descriptions,
                previous_steps=previous_steps,
            )

            if use_tot and step_num > 1 and len(steps_record) >= 2:
                # Use ToT beam search for complex branching points
                thought = self.tot.search(task, strategy="bfs", episode=episode)
            else:
                thought = self.llm.generate(
                    thought_prompt, stop=["\nObservation:"],
                )
            thought = thought.strip()

            episode.log_step("thought", thought)

            # ── Check for Finish ─────────────────────────────────
            finish_match = re.search(r"Finish\[(.+?)\]", thought, re.DOTALL)
            if finish_match:
                answer = finish_match.group(1).strip()
                episode.log_step("finish", answer)
                return answer, step_num, think_count, check_count

            # ── Parse & Execute Action ───────────────────────────
            action_match = re.search(
                r"Action\[(\w+)\]:\s*(.+?)(?:\n|$)", thought, re.DOTALL,
            )
            observation = ""
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()
                episode.log_step("action", f"{tool_name}: {tool_input}")

                observation = self.skills.execute(tool_name, tool_input)
                episode.log_step("observation", observation)

                previous_steps += (
                    f"{thought}\nObservation: {observation}\n\n"
                )
                steps_record.append(f"Action: {tool_name}: {tool_input}")
                steps_record.append(f"Observation: {observation}")
            else:
                previous_steps += f"{thought}\n\n"
                steps_record.append(thought)

            # ── Think Step (Anthropic pattern) ───────────────────
            if observation:
                think_output = self._think_step(
                    observation, plan, steps_record, episode,
                )
                think_count += 1
                previous_steps += f"Think: {think_output}\n\n"

        # Max steps exhausted
        fallback = (
            f"(CompoundReasoner reached max steps ({self.max_steps}). "
            f"Last thought: {thought[:200]})"
        )
        episode.log_step("error", fallback)
        return fallback, step_num, think_count, check_count

    # ═══════════════════════════════════════════════════════════════
    #  Think Step
    # ═══════════════════════════════════════════════════════════════

    def _think_step(
        self,
        observation: str,
        plan: ThinkingResult,
        steps_so_far: List[str],
        episode: "EpisodicLog",
    ) -> str:
        """Analyse latest observation without producing an action.

        Implements the Anthropic "think tool" pattern:
        analyse what the observation means, whether it matches our
        hypothesis, and what to do next.

        Logged as ``type="think"`` — does NOT count toward step limit.
        """
        recent = steps_so_far[-6:]  # last 3 action/observation pairs
        context = "\n".join(recent)
        sub_goals = "\n".join(
            f"  {i+1}. {g}" for i, g in enumerate(plan.sub_goals)
        ) if plan.sub_goals else "(no sub-goals)"

        prompt = (
            "你是一個推理分析器。根據以下最新的工具輸出，進行分析：\n\n"
            f"[最新觀察]\n{observation}\n\n"
            f"[最近推理記錄]\n{context}\n\n"
            f"[計劃子目標]\n{sub_goals}\n\n"
            "請回答：\n"
            "1. 這個結果告訴我什麼？\n"
            "2. 跟我的假設一致嗎？\n"
            "3. 下一步該怎麼做？\n"
        )

        try:
            analysis = self.llm.generate(prompt, max_tokens=256)
        except Exception as exc:
            logger.warning("Think step failed: %s", exc)
            analysis = f"(think step error: {exc})"

        episode.log_step("think", analysis.strip())
        return analysis.strip()

    # ═══════════════════════════════════════════════════════════════
    #  Grounded Self-Check
    # ═══════════════════════════════════════════════════════════════

    def _grounded_self_check(
        self,
        task: str,
        steps_so_far: List[str],
        plan: ThinkingResult,
        episode: "EpisodicLog",
    ) -> SelfCheckResult:
        """Perform a grounded self-check based on external signals.

        Checks:
          1. Tool result consistency with prior reasoning.
          2. KnowledgeStore cross-validation.
          3. Progress toward sub-goals.

        Does NOT rely on LLM self-assessment alone (per Huang et al.,
        ICLR 2024).
        """
        issues: List[str] = []
        completed_goals = 0

        # 1. Check tool result consistency
        observations = [
            s for s in steps_so_far if s.startswith("Observation:")
        ]
        thoughts = [
            s for s in steps_so_far
            if not s.startswith("Observation:") and not s.startswith("Action:")
        ]
        for obs in observations[-3:]:
            for thought in thoughts[-3:]:
                if self._contradicts(obs, thought):
                    issues.append(
                        f"工具結果與推理可能矛盾: "
                        f"{obs[:60]}… vs {thought[:60]}…"
                    )

        # 2. KnowledgeStore cross-validation
        if self.knowledge_store is not None:
            for thought in thoughts[-2:]:
                # Extract key assertions and check
                try:
                    entries = self.knowledge_store.search(thought, top_k=1)
                    if entries:
                        # If KS has relevant info, check for contradiction
                        ks_content = entries[0].content
                        if self._contradicts(ks_content, thought):
                            issues.append(
                                f"推理與知識庫資訊矛盾: {thought[:80]}…"
                            )
                except Exception:
                    pass

        # 3. Progress assessment
        total_goals = len(plan.sub_goals)
        if total_goals > 0:
            steps_text = " ".join(steps_so_far).lower()
            for goal in plan.sub_goals:
                goal_words = set(goal.lower().split())
                if len(goal_words) > 0:
                    overlap = sum(1 for w in goal_words if w in steps_text)
                    if overlap / len(goal_words) > 0.4:
                        completed_goals += 1

        progress = (
            (completed_goals / total_goals * 100)
            if total_goals > 0
            else 50.0
        )

        needs_replan = len(issues) > 0
        passed = not needs_replan

        result = SelfCheckResult(
            passed=passed,
            needs_replan=needs_replan,
            issues=issues,
            progress_pct=round(progress, 1),
        )

        episode.log_step(
            "self_check",
            f"passed={passed}, issues={len(issues)}, "
            f"progress={progress:.0f}%",
        )
        logger.info(
            "Self-check: passed=%s, issues=%d, progress=%.0f%%",
            passed, len(issues), progress,
        )

        return result

    @staticmethod
    def _contradicts(text_a: str, text_b: str) -> bool:
        """Simple contradiction detector using negation heuristic.

        A proper implementation would use embedding cosine distance
        or NLI. This heuristic catches obvious negation patterns.
        """
        a_lower = text_a.lower()
        b_lower = text_b.lower()

        negation_pairs = [
            ("不是", "是"), ("沒有", "有"), ("不能", "能"),
            ("false", "true"), ("incorrect", "correct"),
            ("no", "yes"), ("isn't", "is"), ("不對", "對"),
        ]
        for neg, pos in negation_pairs:
            if (neg in a_lower and pos in b_lower
                    and neg not in b_lower):
                return True
            if (neg in b_lower and pos in a_lower
                    and neg not in a_lower):
                return True
        return False

    # ═══════════════════════════════════════════════════════════════
    #  Phase ③ — Answer Verification
    # ═══════════════════════════════════════════════════════════════

    def _verify_answer(
        self,
        answer: str,
        task: str,
        trace_observations: List[str],
        episode: "EpisodicLog",
    ) -> VerificationResult:
        """Run Answer Verification Gate (HallucinationGuard)."""
        result = self.guard.verify(
            answer, task,
            tool_results=trace_observations,
            episode=episode,
        )
        if result.verdict != VERDICT_PASS:
            episode.log_step(
                "hallucination_detected",
                f"verdict={result.verdict}, "
                f"flagged={result.flagged_claims}, "
                f"score={result.hallucination_score}",
            )
        return result

    @staticmethod
    def _apply_flags(
        answer: str,
        verification: VerificationResult,
    ) -> str:
        """Tag flagged claims with ``[unverified]`` in the answer."""
        if not verification.flagged_claims:
            return answer

        result = answer
        for claim in verification.flagged_claims:
            if claim in result:
                result = result.replace(claim, f"[unverified] {claim}")
        return result

    # ═══════════════════════════════════════════════════════════════
    #  Backward-compatible forced-strategy path
    # ═══════════════════════════════════════════════════════════════

    def _run_forced(
        self,
        task: str,
        episode: "EpisodicLog",
        strategy: str,
    ) -> CompoundResult:
        """Skip Phase ① and run a single strategy directly."""
        episode.log_step(
            "strategy_switch",
            f"Forced strategy: {strategy}",
        )

        if strategy == "cot":
            answer = self.cot.run(task, episode)
        elif strategy == "tot":
            answer = self.tot.search(task, strategy="bfs", episode=episode)
        elif strategy == "react":
            answer = self.react.run(task, episode)
        else:
            logger.warning("Unknown forced strategy '%s', using CoT.", strategy)
            answer = self.cot.run(task, episode)

        return CompoundResult(
            answer=answer,
            confidence=1.0,
            strategy_used=strategy,
            thinking=None,
            verification=None,
            steps_taken=1,
        )

    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _collect_observations(episode: "EpisodicLog") -> List[str]:
        """Extract observation strings from the episode log."""
        observations: List[str] = []
        if hasattr(episode, "steps"):
            for step in episode.steps:
                if step.step_type == "observation" and step.content:
                    observations.append(step.content)
        return observations

    def __repr__(self) -> str:
        return (
            f"CompoundReasoner("
            f"check_interval={self.check_interval}, "
            f"max_steps={self.max_steps})"
        )
