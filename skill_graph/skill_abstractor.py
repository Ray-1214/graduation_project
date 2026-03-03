"""
SkillAbstractor — MDL-based skill extraction from reasoning traces.

Implements the Skill Abstraction Objective:

    min_Σ  L(Σ) + Σᵢ L(τᵢ | Σ)

where L(Σ) is the skill library's description length and L(τᵢ | Σ)
is the cost of encoding trace τᵢ using skills in Σ as macro operators.

A candidate subsequence becomes a new skill only when its compression
gain strictly exceeds the cost of adding it to the library (c_add).

Algorithm
---------
1.  **Canonicalize**: extract the *action* field from each TraceStep to
    form a string sequence per trace.
2.  **N-gram mining**: enumerate all n-grams (n ≥ min_ngram) that appear
    in ≥ 2 traces (or ≥ min_frequency occurrences across all traces).
3.  **Compression gain**: for each candidate subsequence ``s`` of
    length ``k`` occurring ``f`` times:

        gain(s) = (k − 1) × f          # tokens saved when s → macro
        cost(s) = c_add + k             # library entry + definition

    Net gain = gain − cost.  Only candidates with net_gain > 0 pass.
4.  **Greedy selection**: sort candidates by net_gain descending, greedily
    accept non-overlapping candidates.
5.  **SkillNode construction**: each accepted candidate is materialised
    into a ``SkillNode`` with policy = joined action subsequence.

Downstream consumers: EvolutionOperator, SkillGraph.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from memory.episodic_log import EpisodicTrace, TraceStep
from skill_graph.skill_node import SkillNode

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Candidate dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CandidateSkill:
    """A candidate macro-skill extracted from trace subsequences.

    Attributes:
        actions:        The ordered action subsequence.
        frequency:      How many times this subsequence appears.
        length:         Number of actions in the subsequence (len(actions)).
        compression_gain: Raw tokens saved = (length − 1) × frequency.
        description_cost: Cost of adding to library = c_add + length.
        net_gain:       compression_gain − description_cost.
        source_traces:  IDs of traces that contain this subsequence.
    """
    actions: Tuple[str, ...]
    frequency: int
    length: int
    compression_gain: float
    description_cost: float
    net_gain: float
    source_traces: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
#  SkillAbstractor
# ═══════════════════════════════════════════════════════════════════

class SkillAbstractor:
    """Extract reusable macro-skills from a corpus of reasoning traces.

    Uses the Minimum Description Length (MDL) principle:
    a new skill is worth adding only if the compression gain from
    replacing repeated subsequences exceeds the cost of storing
    the skill definition.

    Args:
        c_add:          Fixed cost of adding a skill to the library.
                        Higher → fewer, longer skills accepted.
        min_ngram:      Minimum subsequence length to consider.
        max_ngram:      Maximum subsequence length to consider.
        min_frequency:  Minimum number of occurrences for a candidate.
        max_candidates: Maximum number of candidates to return
                        (after greedy selection).
    """

    def __init__(
        self,
        c_add: float = 2.0,
        min_ngram: int = 2,
        max_ngram: int = 8,
        min_frequency: int = 2,
        max_candidates: int = 20,
    ) -> None:
        self.c_add = c_add
        self.min_ngram = max(min_ngram, 2)   # floor at 2
        self.max_ngram = max_ngram
        self.min_frequency = max(min_frequency, 2)
        self.max_candidates = max_candidates

    # ── Public API ───────────────────────────────────────────────────

    def extract(
        self,
        traces: List[EpisodicTrace],
    ) -> List[CandidateSkill]:
        """Extract candidate skills from a corpus of traces.

        Args:
            traces: List of EpisodicTrace objects.

        Returns:
            Sorted list of CandidateSkill (best net_gain first).
            Only candidates with net_gain > 0 are included.
        """
        if not traces:
            return []

        # 1. Canonicalize — extract action sequences
        action_seqs: List[Tuple[str, ...]] = []
        trace_ids: List[str] = []
        for t in traces:
            seq = tuple(step.action for step in t.steps)
            if len(seq) >= self.min_ngram:
                action_seqs.append(seq)
                trace_ids.append(t.task_id)

        if not action_seqs:
            return []

        # 2. N-gram mining
        ngram_counts, ngram_sources = self._mine_ngrams(
            action_seqs, trace_ids,
        )

        # 3. Compute compression gain for each candidate
        candidates = self._score_candidates(ngram_counts, ngram_sources)

        # 4. Greedy non-overlapping selection
        selected = self._greedy_select(candidates, action_seqs)

        logger.info(
            "SkillAbstractor: %d traces → %d n-grams → %d candidates",
            len(traces), len(ngram_counts), len(selected),
        )
        return selected

    def extract_as_skills(
        self,
        traces: List[EpisodicTrace],
        name_prefix: str = "macro",
    ) -> List[SkillNode]:
        """Extract candidates and materialise them as SkillNode objects.

        Each accepted candidate becomes a SkillNode with:
        - policy: the joined action subsequence
        - initiation_set: derived from source trace descriptions
        - tags: ["abstracted", "phase2"]

        Args:
            traces:      List of EpisodicTrace objects.
            name_prefix: Prefix for generated skill names.

        Returns:
            List of new SkillNode objects (not yet added to graph).
        """
        candidates = self.extract(traces)
        trace_map = {t.task_id: t for t in traces}
        skills: List[SkillNode] = []

        for i, cand in enumerate(candidates):
            # Build policy from action sequence
            policy = " → ".join(cand.actions)

            # Derive initiation set from source trace descriptions
            init_keywords: set[str] = set()
            for tid in cand.source_traces:
                if tid in trace_map:
                    desc = trace_map[tid].task_description.lower()
                    # Extract first 3 significant words
                    words = [w for w in desc.split()
                             if len(w) > 2 and w.isalpha()][:3]
                    init_keywords.update(words)

            # Build termination predicate
            termination = (
                f"Sequence of {cand.length} actions completed: "
                f"{cand.actions[-1]}"
            )

            skill = SkillNode(
                name=f"{name_prefix}_{i+1:03d}",
                policy=policy,
                termination=termination,
                initiation_set=sorted(init_keywords)[:10],
                tags=["abstracted", "phase2"],
            )
            skills.append(skill)

        return skills

    # ── Internal: n-gram mining ──────────────────────────────────────

    def _mine_ngrams(
        self,
        action_seqs: List[Tuple[str, ...]],
        trace_ids: List[str],
    ) -> Tuple[Counter, Dict[Tuple[str, ...], List[str]]]:
        """Extract all n-grams of length [min_ngram, max_ngram].

        Returns:
            ngram_counts:  Counter mapping each n-gram tuple to its
                           total occurrence count across all traces.
            ngram_sources: Dict mapping each n-gram to the list of
                           trace IDs where it appeared.
        """
        ngram_counts: Counter = Counter()
        ngram_sources: Dict[Tuple[str, ...], List[str]] = {}

        for seq, tid in zip(action_seqs, trace_ids):
            seq_len = len(seq)
            seen_in_this_trace: set[Tuple[str, ...]] = set()

            for n in range(self.min_ngram, min(self.max_ngram, seq_len) + 1):
                for start in range(seq_len - n + 1):
                    ngram = seq[start:start + n]
                    ngram_counts[ngram] += 1

                    # Track source traces (dedupe per trace)
                    if ngram not in seen_in_this_trace:
                        seen_in_this_trace.add(ngram)
                        ngram_sources.setdefault(ngram, []).append(tid)

        # Filter by minimum frequency
        ngram_counts = Counter({
            ng: cnt for ng, cnt in ngram_counts.items()
            if cnt >= self.min_frequency
        })

        return ngram_counts, ngram_sources

    # ── Internal: compression scoring ────────────────────────────────

    def _score_candidates(
        self,
        ngram_counts: Counter,
        ngram_sources: Dict[Tuple[str, ...], List[str]],
    ) -> List[CandidateSkill]:
        """Score each n-gram by MDL compression gain.

        gain(s)  = (len(s) − 1) × freq(s)
                   (each occurrence saves len(s)−1 tokens by replacing
                    the subsequence with a single macro call)

        cost(s)  = c_add + len(s)
                   (fixed library cost + definition length)

        net_gain = gain − cost
        """
        candidates: List[CandidateSkill] = []

        for ngram, freq in ngram_counts.items():
            k = len(ngram)
            gain = (k - 1) * freq
            cost = self.c_add + k
            net = gain - cost

            if net > 0:
                candidates.append(CandidateSkill(
                    actions=ngram,
                    frequency=freq,
                    length=k,
                    compression_gain=gain,
                    description_cost=cost,
                    net_gain=net,
                    source_traces=ngram_sources.get(ngram, []),
                ))

        # Sort by net_gain descending, then by length descending (prefer longer)
        candidates.sort(key=lambda c: (-c.net_gain, -c.length))
        return candidates

    # ── Internal: greedy non-overlapping selection ────────────────────

    def _greedy_select(
        self,
        candidates: List[CandidateSkill],
        action_seqs: List[Tuple[str, ...]],
    ) -> List[CandidateSkill]:
        """Greedily select candidates, skipping those whose action tuple
        is a strict subsequence of an already-selected longer candidate.

        This prevents redundancy: if (A, B, C) is selected, we skip (A, B).
        """
        selected: List[CandidateSkill] = []
        accepted_grams: set[Tuple[str, ...]] = set()

        for cand in candidates:
            if len(selected) >= self.max_candidates:
                break

            # Skip if this n-gram is a strict sub-tuple of one already accepted
            if self._is_subsumed(cand.actions, accepted_grams):
                continue

            selected.append(cand)
            accepted_grams.add(cand.actions)

        return selected

    @staticmethod
    def _is_subsumed(
        ngram: Tuple[str, ...],
        accepted: set[Tuple[str, ...]],
    ) -> bool:
        """Check if ngram is a contiguous sub-sequence of any accepted ngram."""
        for existing in accepted:
            if len(existing) <= len(ngram):
                continue
            # Check if ngram appears as contiguous slice of existing
            elen = len(existing)
            nlen = len(ngram)
            for i in range(elen - nlen + 1):
                if existing[i:i + nlen] == ngram:
                    return True
        return False

    # ── Display ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SkillAbstractor(c_add={self.c_add}, "
            f"ngram=[{self.min_ngram},{self.max_ngram}], "
            f"min_freq={self.min_frequency})"
        )
