"""
EvolutionOperator Φ — maps (G_t, τ_t) → G_{t+1}.

The evolution operator is the core mechanism that drives the
Self-Evolving Skill Graph forward.  It is composed of four
sequential sub-operations:

    (i)   Utility evaluation  — decay + reinforcement
    (ii)  Skill insertion     — MDL-based abstraction + dedup
    (iii) Subgraph contraction — merge co-occurring subsequences
    (iv)  Memory tier update  — reassign active/cold/archive

If none of the three trigger conditions (T1–T3) fire, Φ degenerates
to step (i) only (utility decay).

Trigger conditions
------------------
  T1: Task failure       (trace.success == False)
  T2: Trace too long     (len(trace) > moving_avg + η)
  T3: Repeated sub-seq   (a subsequence occurs > δ times in history)
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from memory.episodic_log import EpisodicTrace, TraceStep
from skill_graph.contract_subgraph import contract_subgraph
from skill_graph.memory_partition import MemoryPartition
from skill_graph.skill_abstractor import SkillAbstractor
from skill_graph.skill_graph import SkillGraph
from skill_graph.skill_node import SkillNode

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  EvolutionLog — record of what happened during one Φ application
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EvolutionLog:
    """Record of actions taken by a single :meth:`evolve` call.

    Attributes:
        timestamp:         When the evolution started.
        triggers_fired:    Which trigger conditions were active (T1/T2/T3).
        decayed_skills:    Number of skills that received utility decay.
        reinforced_skills: IDs of skills that received reinforcement.
        inserted_skills:   IDs of newly inserted skills.
        rejected_skills:   IDs of candidates that failed dedup/validation.
        contracted:        List of (sequence, macro_id) tuples.
        tier_changes:      Dict of skill_id → (old_tier, new_tier).
        full_evolution:    Whether steps (ii)–(iii) ran (vs decay-only).
    """
    timestamp: float = field(default_factory=time.time)
    triggers_fired: List[str] = field(default_factory=list)
    decayed_skills: int = 0
    reinforced_skills: List[str] = field(default_factory=list)
    inserted_skills: List[str] = field(default_factory=list)
    rejected_skills: List[str] = field(default_factory=list)
    contracted: List[Dict[str, Any]] = field(default_factory=list)
    tier_changes: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    full_evolution: bool = False

    def summary(self) -> str:
        parts = [f"Φ @ {time.strftime('%H:%M:%S', time.localtime(self.timestamp))}"]
        if self.triggers_fired:
            parts.append(f"triggers={self.triggers_fired}")
        parts.append(f"decay={self.decayed_skills}")
        if self.reinforced_skills:
            parts.append(f"reinforced={len(self.reinforced_skills)}")
        if self.inserted_skills:
            parts.append(f"inserted={len(self.inserted_skills)}")
        if self.contracted:
            parts.append(f"contracted={len(self.contracted)}")
        tier_moved = sum(1 for o, n in self.tier_changes.values() if o != n)
        if tier_moved:
            parts.append(f"tier_moves={tier_moved}")
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════
#  EvolutionOperator
# ═══════════════════════════════════════════════════════════════════

class EvolutionOperator:
    """Maps (G_t, τ_t) → G_{t+1} via utility evaluation, skill
    insertion, subgraph contraction, and memory tier updates.

    Args:
        gamma:          Utility decay rate γ ∈ (0, 1).
        delta_u:        Default reinforcement ΔU for used skills.
        theta_dup:      Deduplication similarity threshold.
                        Candidate rejected if sim ≥ θ_dup.
        delta:          Co-occurrence frequency threshold for
                        subgraph contraction.
        eta:            Trace-length anomaly margin.
                        T2 fires when len(τ) > avg + η.
        abstractor:     Optional pre-configured SkillAbstractor.
        trace_history:  Shared trace buffer for cross-episode analysis.
    """

    def __init__(
        self,
        gamma: float = 0.05,
        delta_u: float = 1.0,
        theta_dup: float = 0.85,
        delta: int = 3,
        eta: int = 5,
        abstractor: Optional[SkillAbstractor] = None,
        trace_history: Optional[List[EpisodicTrace]] = None,
    ) -> None:
        self.gamma = gamma
        self.delta_u = delta_u
        self.theta_dup = theta_dup
        self.delta = delta
        self.eta = eta
        self.abstractor = abstractor or SkillAbstractor()
        self.trace_history: List[EpisodicTrace] = trace_history or []

        # Running stats for T2 trigger
        self._trace_lengths: List[int] = []

    # ── Public API ───────────────────────────────────────────────────

    def evolve(
        self,
        graph: SkillGraph,
        trace: EpisodicTrace,
        partition: MemoryPartition,
    ) -> EvolutionLog:
        """Apply one evolution step: Φ(G_t, τ_t) → G_{t+1}.

        Args:
            graph:     The skill graph to evolve **in-place**.
            trace:     This episode's reasoning trace.
            partition: Memory partition manager.

        Returns:
            EvolutionLog recording all actions taken.
        """
        log = EvolutionLog()

        # Record trace for history
        self.trace_history.append(trace)
        self._trace_lengths.append(len(trace))

        # ── Check trigger conditions ─────────────────────────────
        triggers = self._check_triggers(trace)
        log.triggers_fired = triggers
        do_full = len(triggers) > 0
        log.full_evolution = do_full

        # ── Step 1: Utility evaluation (always runs) ─────────────
        self._step_utility_evaluation(graph, trace, log)

        # ── Steps 2–3: only if triggered ─────────────────────────
        if do_full:
            # Step 2: Skill insertion
            self._step_skill_insertion(graph, log)

            # Step 3: Subgraph contraction
            self._step_subgraph_contraction(graph, trace, log)

        # ── Step 4: Memory tier update (always runs) ─────────────
        self._step_memory_tier_update(graph, partition, log)

        logger.info("EvolutionOperator: %s", log.summary())
        return log

    # ── Trigger conditions ───────────────────────────────────────────

    def _check_triggers(self, trace: EpisodicTrace) -> List[str]:
        """Evaluate T1, T2, T3 and return list of fired trigger names."""
        fired: List[str] = []

        # T1: Task failure
        if not trace.success:
            fired.append("T1_failure")

        # T2: Trace too long (len > moving_avg + η)
        if len(self._trace_lengths) >= 2:
            avg = sum(self._trace_lengths[:-1]) / len(self._trace_lengths[:-1])
            if len(trace) > avg + self.eta:
                fired.append("T2_long_trace")

        # T3: Repeated sub-sequence frequency > δ
        if self._has_repeated_subsequence(trace):
            fired.append("T3_repeated_subseq")

        return fired

    def _has_repeated_subsequence(self, trace: EpisodicTrace) -> bool:
        """Check if any action sub-sequence of length ≥ 2 appears > δ
        times across the trace history (including current trace)."""
        # Build action sequences from recent history
        actions_corpus: List[Tuple[str, ...]] = []
        for t in self.trace_history[-20:]:  # look at last 20 traces
            actions_corpus.append(tuple(step.action for step in t.steps))

        if len(actions_corpus) < 2:
            return False

        # Count 2-grams across corpus
        bigram_counts: Counter = Counter()
        for seq in actions_corpus:
            for i in range(len(seq) - 1):
                bigram_counts[seq[i:i + 2]] += 1

        return any(cnt > self.delta for cnt in bigram_counts.values())

    # ── Step 1: Utility evaluation ───────────────────────────────────

    def _step_utility_evaluation(
        self,
        graph: SkillGraph,
        trace: EpisodicTrace,
        log: EvolutionLog,
    ) -> None:
        """Decay all skills, then reinforce those used in this trace."""
        # Decay
        graph.decay_all(gamma=self.gamma)
        log.decayed_skills = len(graph)

        # Identify skills used in this trace (match by action → skill name)
        trace_actions = {step.action for step in trace.steps}
        for skill in graph.skills:
            # Match by name, policy keyword, or initiation set
            if (skill.name in trace_actions
                    or skill.policy in trace_actions
                    or any(a in skill.policy for a in trace_actions)):
                # Compute reinforcement proportional to trace score
                delta = self.delta_u * max(trace.score, 0.1)
                cost = trace.total_time / max(len(trace), 1)
                skill.reinforce(delta_u=delta, cost=cost)
                log.reinforced_skills.append(skill.skill_id)

    # ── Step 2: Skill insertion ──────────────────────────────────────

    def _step_skill_insertion(
        self,
        graph: SkillGraph,
        log: EvolutionLog,
    ) -> None:
        """Use SkillAbstractor on trace history, validate + dedup, insert."""
        if len(self.trace_history) < 2:
            return  # need at least 2 traces for pattern mining

        # Extract candidates
        candidate_skills = self.abstractor.extract_as_skills(
            self.trace_history[-20:],  # use recent history window
        )

        for skill in candidate_skills:
            # (a) Structural validation
            if not skill.policy or not skill.policy.strip():
                log.rejected_skills.append(skill.skill_id)
                continue

            # (b) Deduplication check
            if self._is_duplicate(skill, graph):
                log.rejected_skills.append(skill.skill_id)
                continue

            # (c) Capacity check + insert
            try:
                graph.add_skill(skill)
                log.inserted_skills.append(skill.skill_id)
                logger.info(
                    "Inserted new skill: %s (policy=%s)",
                    skill.skill_id, skill.policy[:60],
                )
            except (ValueError, OverflowError) as exc:
                logger.warning("Skill insertion failed: %s", exc)
                log.rejected_skills.append(skill.skill_id)

    def _is_duplicate(self, candidate: SkillNode, graph: SkillGraph) -> bool:
        """Check if candidate's policy is too similar to any existing skill."""
        for existing in graph.skills:
            sim = SequenceMatcher(
                None, candidate.policy, existing.policy,
            ).ratio()
            if sim >= self.theta_dup:
                logger.debug(
                    "Dedup: '%s' too similar to '%s' (sim=%.2f ≥ %.2f)",
                    candidate.name, existing.name, sim, self.theta_dup,
                )
                return True
        return False

    # ── Step 3: Subgraph contraction ─────────────────────────────────

    def _step_subgraph_contraction(
        self,
        graph: SkillGraph,
        trace: EpisodicTrace,
        log: EvolutionLog,
    ) -> None:
        """Find co-occurring skill subsequences and contract them."""
        # Find action pairs that appear frequently enough
        recent_traces = self.trace_history[-20:]
        pair_counts: Counter = Counter()
        for t in recent_traces:
            actions = [step.action for step in t.steps]
            for i in range(len(actions) - 1):
                pair_counts[(actions[i], actions[i + 1])] += 1

        # For each frequent pair, check if both exist as skills in graph
        contracted_this_round: Set[str] = set()
        for (a1, a2), count in pair_counts.most_common():
            if count <= self.delta:
                break  # sorted by frequency, can stop early

            # Find matching skill IDs
            sid1 = self._find_skill_by_action(graph, a1, contracted_this_round)
            sid2 = self._find_skill_by_action(graph, a2, contracted_this_round)

            if sid1 is None or sid2 is None or sid1 == sid2:
                continue

            # Both skills exist → contract
            macro_id = f"macro-{sid1}-{sid2}"
            if graph.has_skill(macro_id):
                continue

            try:
                macro = contract_subgraph(
                    graph, [sid1, sid2], macro_id,
                    name=f"macro({a1},{a2})",
                )
                contracted_this_round.add(sid1)
                contracted_this_round.add(sid2)
                log.contracted.append({
                    "sequence": [sid1, sid2],
                    "macro_id": macro_id,
                    "frequency": count,
                })
                logger.info(
                    "Contracted [%s, %s] → %s (freq=%d)",
                    sid1, sid2, macro_id, count,
                )
            except (ValueError, KeyError) as exc:
                logger.debug("Contraction skipped: %s", exc)

    @staticmethod
    def _find_skill_by_action(
        graph: SkillGraph,
        action: str,
        exclude: Set[str],
    ) -> Optional[str]:
        """Find a skill whose name or policy matches the action string."""
        for skill in graph.skills:
            if skill.skill_id in exclude:
                continue
            if skill.name == action or action in skill.policy:
                return skill.skill_id
        return None

    # ── Step 4: Memory tier update ───────────────────────────────────

    def _step_memory_tier_update(
        self,
        graph: SkillGraph,
        partition: MemoryPartition,
        log: EvolutionLog,
    ) -> None:
        """Reassign all skill tiers via MemoryPartition."""
        old_tiers = {
            skill.skill_id: partition.get_tier(skill.skill_id)
            for skill in graph.skills
        }
        new_tiers = partition.update_all(graph)

        for sid, new_tier in new_tiers.items():
            old_tier = old_tiers.get(sid, "cold")
            if old_tier != new_tier:
                log.tier_changes[sid] = (old_tier, new_tier)

    # ── Display ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"EvolutionOperator(γ={self.gamma}, ΔU={self.delta_u}, "
            f"θ_dup={self.theta_dup}, δ={self.delta}, η={self.eta}, "
            f"history={len(self.trace_history)})"
        )
