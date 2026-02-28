"""
Long-term memory — persistent JSON store for reflexion lessons.

Stores structured reflection entries (task, reflection, lessons,
score, timestamp) and supports keyword-based retrieval.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReflectionEntry:
    """A single reflexion memory."""
    task: str
    reflection: str
    lessons: List[str]
    score: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReflectionEntry":
        return cls(**d)


class LongTermMemory:
    """JSON-file backed persistent store for reflexion memories."""

    def __init__(self, store_path: str) -> None:
        self.store_path = Path(store_path)
        self._entries: List[ReflectionEntry] = []
        self._load()

    # -- persistence --

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text())
                self._entries = [ReflectionEntry.from_dict(d) for d in data]
                logger.info("Loaded %d reflection entries.", len(self._entries))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Corrupt long-term memory file, starting fresh: %s", exc)
                self._entries = []
        else:
            self._entries = []

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps([e.to_dict() for e in self._entries], indent=2)
        )

    # -- public API --

    def store(self, entry: ReflectionEntry) -> None:
        """Persist a new reflection entry."""
        self._entries.append(entry)
        self._save()
        logger.info("Stored reflection for task: %s", entry.task[:60])

    def retrieve(self, query_keywords: List[str], top_k: int = 3) -> List[ReflectionEntry]:
        """Simple keyword-based retrieval (case-insensitive).

        Returns the top_k entries that match the most keywords.
        """
        scored: List[tuple[int, ReflectionEntry]] = []
        query_lower = [kw.lower() for kw in query_keywords]
        for entry in self._entries:
            text = (entry.task + " " + entry.reflection + " " + " ".join(entry.lessons)).lower()
            score = sum(1 for kw in query_lower if kw in text)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def all(self) -> List[ReflectionEntry]:
        """Return all stored entries."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"LongTermMemory(entries={len(self)}, path={self.store_path})"
