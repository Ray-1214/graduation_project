"""
Short-term (working) memory — ring buffer of recent messages.

Used to keep a sliding window of conversation context that can be
injected into prompts without exceeding the model's context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional
import time


@dataclass
class Message:
    """A single entry in working memory."""
    role: Literal["system", "user", "assistant", "observation"]
    content: str
    timestamp: float = field(default_factory=time.time)


class ShortTermMemory:
    """Fixed-capacity ring buffer of Message objects."""

    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self._buffer: List[Message] = []

    # -- mutators --

    def add(self, role: str, content: str) -> None:
        """Append a message; evict the oldest if at capacity."""
        self._buffer.append(Message(role=role, content=content))  # type: ignore[arg-type]
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def clear(self) -> None:
        self._buffer.clear()

    # -- accessors --

    def get_context(self, last_n: Optional[int] = None) -> str:
        """Return the last N messages formatted as a string."""
        msgs = self._buffer if last_n is None else self._buffer[-last_n:]
        return "\n".join(f"[{m.role}] {m.content}" for m in msgs)

    def get_messages(self, last_n: Optional[int] = None) -> List[Message]:
        """Return raw Message objects."""
        return list(self._buffer if last_n is None else self._buffer[-last_n:])

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ShortTermMemory(len={len(self)}, cap={self.capacity})"
