"""
Episodic log — structured trace of a single reasoning episode.

Records each step (thought, action, observation, evaluation, etc.)
with timestamps. Used by Reflexion and experiment logging.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


StepType = Literal[
    "thought", "action", "observation", "evaluation",
    "reflection", "branch", "finish", "error",
]


@dataclass
class EpisodeStep:
    """A single step in a reasoning episode."""
    step_type: StepType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodicLog:
    """Full trace of one reasoning episode."""

    task: str
    strategy: str
    steps: List[EpisodeStep] = field(default_factory=list)
    result: Optional[str] = None
    score: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # -- recording --

    def log_step(self, step_type: StepType, content: str, **metadata: Any) -> None:
        self.steps.append(EpisodeStep(
            step_type=step_type,
            content=content,
            metadata=metadata,
        ))

    def finish(self, result: str, score: Optional[float] = None) -> None:
        self.result = result
        self.score = score
        self.end_time = time.time()

    # -- serialization --

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "strategy": self.strategy,
            "result": self.result,
            "score": self.score,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": (self.end_time - self.start_time) if self.end_time else None,
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    def trajectory_str(self) -> str:
        """Render the episode as a human-readable trajectory string."""
        lines = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"[{step.step_type.upper()}] {step.content}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return (
            f"EpisodicLog(task={self.task[:40]!r}, "
            f"strategy={self.strategy}, steps={len(self)})"
        )


# =========================================================================
# Canonical Trace Format (for Skill Graph downstream modules)
# =========================================================================

@dataclass
class TraceStep:
    """A single (state → action → outcome) triple in a canonical trace.

    Consumed by: SkillAbstractor, EvolutionOperator, MetricsTracker.
    """
    state: str        # current context (task description or prior outcome)
    action: str       # agent's action (reasoning step, tool call, etc.)
    outcome: str      # result of the action (LLM response, tool output, etc.)
    timestamp: float  # epoch timestamp


@dataclass
class EpisodicTrace:
    """Standardised trace of a complete reasoning episode.

    This is the canonical interchange format — all downstream
    modules (SkillAbstractor, EvolutionOperator, MetricsTracker)
    depend on this structure.
    """
    task_id: str
    task_description: str
    steps: List[TraceStep]
    strategy: str         # "cot" / "tot" / "react" / "reflexion"
    success: bool
    score: float          # 0.0 – 1.0
    total_time: float     # seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "strategy": self.strategy,
            "success": self.success,
            "score": self.score,
            "total_time": self.total_time,
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return (
            f"EpisodicTrace(task_id={self.task_id!r}, "
            f"strategy={self.strategy}, steps={len(self)}, "
            f"success={self.success}, score={self.score:.2f})"
        )


# -- Step-type grouping for conversion ------------------------------------

# Step types that represent an *action* taken by the agent
_ACTION_TYPES = {"thought", "action", "branch", "reflection"}
# Step types that represent the *outcome* of a preceding action
_OUTCOME_TYPES = {"observation", "evaluation", "finish", "error"}


def convert_log_to_trace(
    log: EpisodicLog,
    task_id: Optional[str] = None,
    success_threshold: float = 0.5,
) -> EpisodicTrace:
    """Convert an EpisodicLog into the canonical EpisodicTrace format.

    Pairs consecutive steps into (state, action, outcome) triples:
      - *action* steps (thought, action, branch, reflection) open a new triple.
      - *outcome* steps (observation, evaluation, finish, error) close it.
      - The *state* is the task description for the first triple, then the
        previous outcome for subsequent triples.

    Args:
        log: A completed EpisodicLog instance.
        task_id: Optional explicit task ID.  If omitted a deterministic
                 ID is generated from the task text and start time.
        success_threshold: Score at or above this value counts as success.

    Returns:
        An EpisodicTrace ready for downstream consumption.
    """
    import hashlib

    # Generate a deterministic task_id if not provided
    if task_id is None:
        raw = f"{log.task}:{log.start_time}"
        task_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    trace_steps: List[TraceStep] = []
    state = f"Task: {log.task}"  # initial state is the task itself

    # Walk through episode steps and pair action→outcome
    pending_action: Optional[EpisodeStep] = None

    for step in log.steps:
        if step.step_type in _ACTION_TYPES:
            # If there's already a pending action with no outcome, flush it
            if pending_action is not None:
                trace_steps.append(TraceStep(
                    state=state,
                    action=f"[{pending_action.step_type}] {pending_action.content}",
                    outcome="(no explicit outcome)",
                    timestamp=pending_action.timestamp,
                ))
                state = pending_action.content

            pending_action = step

        elif step.step_type in _OUTCOME_TYPES:
            action_str = (
                f"[{pending_action.step_type}] {pending_action.content}"
                if pending_action
                else "(implicit)"
            )
            ts = pending_action.timestamp if pending_action else step.timestamp

            trace_steps.append(TraceStep(
                state=state,
                action=action_str,
                outcome=f"[{step.step_type}] {step.content}",
                timestamp=ts,
            ))
            state = step.content
            pending_action = None

    # Flush any remaining pending action
    if pending_action is not None:
        trace_steps.append(TraceStep(
            state=state,
            action=f"[{pending_action.step_type}] {pending_action.content}",
            outcome=log.result or "(no outcome)",
            timestamp=pending_action.timestamp,
        ))

    score = log.score if log.score is not None else 0.0
    total_time = (
        (log.end_time - log.start_time) if log.end_time else 0.0
    )

    return EpisodicTrace(
        task_id=task_id,
        task_description=log.task,
        steps=trace_steps,
        strategy=log.strategy,
        success=score >= success_threshold,
        score=score,
        total_time=round(total_time, 3),
    )
