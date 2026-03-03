"""
Tests for SkillAbstractor — MDL-based skill extraction.

Covers:
  - Basic n-gram extraction from repeated traces
  - Compression gain filtering (only net_gain > 0 accepted)
  - Greedy subsumption (shorter n-gram skipped when longer accepted)
  - extract_as_skills materialises SkillNode objects
  - Edge cases: empty traces, single trace, no repetition
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.episodic_log import EpisodicTrace, TraceStep
from skill_graph.skill_abstractor import CandidateSkill, SkillAbstractor
from skill_graph.skill_node import SkillNode


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_trace(
    task_id: str,
    actions: list[str],
    description: str = "test task",
) -> EpisodicTrace:
    """Build an EpisodicTrace from a list of action strings."""
    steps = [
        TraceStep(
            state=f"state_{i}",
            action=a,
            outcome=f"outcome_{i}",
            timestamp=time.time(),
        )
        for i, a in enumerate(actions)
    ]
    return EpisodicTrace(
        task_id=task_id,
        task_description=description,
        steps=steps,
        strategy="react",
        success=True,
        score=0.8,
        total_time=1.0,
    )


# ═══════════════════════════════════════════════════════════════════
#  Test Cases
# ═══════════════════════════════════════════════════════════════════

class TestSkillAbstractor:

    def test_basic_extraction(self):
        """Repeated action subsequence is detected with positive net gain."""
        # Pattern (search, read, summarize) appears in 4 traces → should be extracted
        traces = [
            _make_trace("t1", ["search", "read", "summarize", "answer"]),
            _make_trace("t2", ["search", "read", "summarize", "verify"]),
            _make_trace("t3", ["search", "read", "summarize", "respond"]),
            _make_trace("t4", ["search", "read", "summarize", "submit"]),
        ]
        sa = SkillAbstractor(c_add=2.0, min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)

        assert len(candidates) >= 1
        # The (search, read, summarize) pattern should be among candidates
        action_sets = [c.actions for c in candidates]
        assert ("search", "read", "summarize") in action_sets

        # Verify the best candidate has positive net gain
        best = candidates[0]
        assert best.net_gain > 0
        assert best.frequency >= 4

    def test_compression_gain_formula(self):
        """Verify gain = (k-1)*f and cost = c_add + k."""
        traces = [
            _make_trace("t1", ["A", "B", "C"]),
            _make_trace("t2", ["A", "B", "C"]),
            _make_trace("t3", ["A", "B", "C"]),
        ]
        # (A,B,C): k=3, f=3 → gain = 2*3 = 6, cost = 2+3 = 5, net = 1
        sa = SkillAbstractor(c_add=2.0, min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)

        abc = [c for c in candidates if c.actions == ("A", "B", "C")]
        assert len(abc) == 1
        c = abc[0]
        assert c.length == 3
        assert c.frequency == 3
        assert c.compression_gain == (3 - 1) * 3  # 6
        assert c.description_cost == 2.0 + 3       # 5
        assert c.net_gain == 1.0

    def test_below_threshold_rejected(self):
        """Candidate with net_gain ≤ 0 is excluded."""
        traces = [
            _make_trace("t1", ["X", "Y"]),
            _make_trace("t2", ["X", "Y"]),
        ]
        # (X,Y): k=2, f=2 → gain = 1*2 = 2, cost = 10+2 = 12, net = -10
        sa = SkillAbstractor(c_add=10.0, min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)

        assert len(candidates) == 0  # all rejected

    def test_greedy_subsumption(self):
        """Shorter n-gram is skipped when a longer containing n-gram is accepted."""
        traces = [
            _make_trace("t1", ["A", "B", "C", "D"]),
            _make_trace("t2", ["A", "B", "C", "D"]),
            _make_trace("t3", ["A", "B", "C", "D"]),
            _make_trace("t4", ["A", "B", "C", "D"]),
            _make_trace("t5", ["A", "B", "C", "D"]),
        ]
        sa = SkillAbstractor(c_add=1.0, min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)

        accepted_actions = [c.actions for c in candidates]
        # If (A,B,C,D) is accepted, (A,B,C) should be subsumed
        if ("A", "B", "C", "D") in accepted_actions:
            assert ("A", "B", "C") not in accepted_actions
            assert ("B", "C", "D") not in accepted_actions

    def test_extract_as_skills(self):
        """extract_as_skills returns SkillNode objects with proper fields."""
        traces = [
            _make_trace("t1", ["plan", "execute", "verify"], "math problem solving"),
            _make_trace("t2", ["plan", "execute", "verify"], "code generation task"),
            _make_trace("t3", ["plan", "execute", "verify"], "data analysis work"),
        ]
        sa = SkillAbstractor(c_add=1.0, min_ngram=2, min_frequency=2)
        skills = sa.extract_as_skills(traces, name_prefix="test")

        assert len(skills) >= 1

        skill = skills[0]
        assert isinstance(skill, SkillNode)
        assert skill.name.startswith("test_")
        assert "plan" in skill.policy
        assert "abstracted" in skill.tags
        assert "phase2" in skill.tags
        assert skill.version == 1
        assert skill.frequency == 0  # fresh skill, never used yet

    def test_source_traces_tracked(self):
        """Candidates record which traces they appeared in."""
        traces = [
            _make_trace("trace_A", ["go", "fetch", "return"]),
            _make_trace("trace_B", ["go", "fetch", "return"]),
            _make_trace("trace_C", ["sit", "stay", "good_boy"]),
        ]
        sa = SkillAbstractor(c_add=1.0, min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)

        gfr = [c for c in candidates
               if c.actions == ("go", "fetch", "return")]
        if gfr:
            assert "trace_A" in gfr[0].source_traces
            assert "trace_B" in gfr[0].source_traces
            assert "trace_C" not in gfr[0].source_traces

    def test_empty_traces(self):
        """Empty input → empty candidates, no crash."""
        sa = SkillAbstractor()
        assert sa.extract([]) == []

    def test_single_trace(self):
        """Single trace → no candidates (min_frequency=2)."""
        traces = [_make_trace("t1", ["A", "B", "C"])]
        sa = SkillAbstractor(min_frequency=2)
        assert sa.extract(traces) == []

    def test_no_repetition(self):
        """All unique actions → no candidates."""
        traces = [
            _make_trace("t1", ["A", "B", "C"]),
            _make_trace("t2", ["D", "E", "F"]),
            _make_trace("t3", ["G", "H", "I"]),
        ]
        sa = SkillAbstractor(min_frequency=2)
        assert sa.extract(traces) == []

    def test_min_ngram_enforced(self):
        """Single-action "n-grams" are never extracted (min_ngram≥2)."""
        traces = [
            _make_trace("t1", ["A", "B"]),
            _make_trace("t2", ["A", "C"]),
        ]
        sa = SkillAbstractor(min_ngram=2, min_frequency=2)
        candidates = sa.extract(traces)
        for c in candidates:
            assert c.length >= 2

    def test_max_candidates_limit(self):
        """max_candidates caps the output length."""
        # Generate many repeating patterns
        traces = []
        for i in range(20):
            traces.append(_make_trace(
                f"t{i}",
                [f"step{j}" for j in range(10)]
            ))
        sa = SkillAbstractor(c_add=0.1, min_frequency=2, max_candidates=3)
        candidates = sa.extract(traces)
        assert len(candidates) <= 3

    def test_repr(self):
        """__repr__ returns readable string."""
        sa = SkillAbstractor(c_add=3.0, min_ngram=3, max_ngram=6, min_frequency=4)
        r = repr(sa)
        assert "c_add=3.0" in r
        assert "ngram=[3,6]" in r
        assert "min_freq=4" in r
