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
