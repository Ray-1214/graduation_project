"""
KnowledgeStore — structured knowledge repository with semantic search.

Combines **JSON persistence** (for durability) with an optional
**VectorStore** (for semantic similarity retrieval).

Each piece of knowledge is a :class:`KnowledgeEntry` carrying source
attribution (``"web"`` / ``"admin"`` / ``"rag"``), confidence, and
usage tracking.

Usage flow in the ReAct loop::

    store = KnowledgeStore()

    # Before calling web_search, check if we already know the answer
    if store.has_knowledge(query, threshold=0.8):
        results = store.search(query, top_k=3)
        ...  # use local knowledge
    else:
        ...  # fall back to web_search
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = (
    Path(__file__).resolve().parent / "knowledge_base" / "knowledge_store.json"
)

# Confidence defaults per source
_DEFAULT_CONFIDENCE = {
    "web": 0.6,
    "admin": 0.9,
    "rag": 0.7,
}


# ─────────────────────────────────────────────────────────────────────
#  KnowledgeEntry
# ─────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeEntry:
    """A single piece of stored knowledge.

    Attributes:
        entry_id:   Unique identifier.
        query:      The original question / search query.
        content:    The knowledge content (answer / snippet).
        source:     Where it came from: ``"web"`` | ``"admin"`` | ``"rag"``.
        confidence: Confidence score ∈ [0, 1].
        timestamp:  Unix timestamp when stored.
        url:        Source URL (web results only).
        used_count: How many times this entry has been retrieved.
    """

    query: str
    content: str
    source: str = "web"
    confidence: float = 0.6
    timestamp: float = field(default_factory=time.time)
    url: Optional[str] = None
    used_count: int = 0
    entry_id: str = field(default_factory=lambda: f"ke-{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────
#  KnowledgeStore
# ─────────────────────────────────────────────────────────────────────

class KnowledgeStore:
    """Structured knowledge repository with semantic search.

    Args:
        store_path:  Path to the JSON persistence file.
        use_vectors: If ``True``, maintain a VectorStore index for
                     semantic similarity search.  Requires
                     ``sentence-transformers`` and ``faiss-cpu``.
        max_age:     Staleness threshold in seconds (default: 7 days).
                     Entries older than this are de-prioritised but
                     not deleted.
    """

    def __init__(
        self,
        store_path: Optional[Path] = None,
        use_vectors: bool = True,
        max_age: float = 604800.0,  # 7 days
    ) -> None:
        self.store_path = Path(store_path) if store_path else _DEFAULT_STORE_PATH
        self.max_age = max_age
        self._use_vectors = use_vectors

        # Internal state
        self._entries: Dict[str, KnowledgeEntry] = {}  # entry_id → entry
        self._vector_store = None
        self._vector_ids: List[str] = []  # parallel list with VectorStore texts

        # Load persisted data
        self._load()

    # ── VectorStore (lazy) ───────────────────────────────────────────

    def _ensure_vector_store(self) -> bool:
        """Lazily init the VectorStore.  Returns True on success."""
        if self._vector_store is not None:
            return True
        if not self._use_vectors:
            return False
        try:
            from memory.vector_store import VectorStore
            self._vector_store = VectorStore()
            # Re-index existing entries
            if self._entries:
                texts = []
                ids = []
                for eid, entry in self._entries.items():
                    texts.append(f"{entry.query} {entry.content}")
                    ids.append(eid)
                self._vector_store.add_texts(
                    texts,
                    metadatas=[{"entry_id": eid} for eid in ids],
                )
                self._vector_ids = ids
                logger.info("VectorStore indexed %d entries", len(ids))
            return True
        except Exception as exc:
            logger.warning("VectorStore unavailable: %s", exc)
            self._use_vectors = False
            return False

    # ── Public API ───────────────────────────────────────────────────

    def store(self, entry: KnowledgeEntry) -> str:
        """Store a knowledge entry and return its ID.

        Persists to JSON and (if available) indexes in VectorStore.
        """
        self._entries[entry.entry_id] = entry

        # Index in VectorStore
        if self._ensure_vector_store():
            text = f"{entry.query} {entry.content}"
            self._vector_store.add_texts(
                [text],
                metadatas=[{"entry_id": entry.entry_id}],
            )
            self._vector_ids.append(entry.entry_id)

        self._save()
        logger.info(
            "Stored knowledge [%s] source=%s conf=%.2f: %s",
            entry.entry_id, entry.source, entry.confidence,
            entry.query[:60],
        )
        return entry.entry_id

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeEntry]:
        """Semantic search for relevant knowledge entries.

        Uses VectorStore embeddings when available, otherwise falls
        back to keyword matching.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of matching :class:`KnowledgeEntry` objects, sorted
            by relevance. ``used_count`` is incremented for each
            returned entry.
        """
        results: List[KnowledgeEntry] = []

        if self._ensure_vector_store() and self._vector_store.size > 0:
            # Semantic search
            docs = self._vector_store.query(query, top_k=top_k)
            for doc in docs:
                eid = doc.metadata.get("entry_id")
                entry = self._entries.get(eid)
                if entry is not None:
                    entry.used_count += 1
                    results.append(entry)
        else:
            # Fallback: keyword matching
            q_lower = query.lower()
            scored = []
            for entry in self._entries.values():
                score = 0.0
                if q_lower in entry.query.lower():
                    score = 0.9
                elif q_lower in entry.content.lower():
                    score = 0.6
                elif any(w in entry.query.lower() for w in q_lower.split()):
                    score = 0.4
                if score > 0:
                    scored.append((score, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, entry in scored[:top_k]:
                entry.used_count += 1
                results.append(entry)

        if results:
            self._save()  # persist used_count updates
        return results

    def has_knowledge(self, query: str, threshold: float = 0.8) -> bool:
        """Check whether sufficiently relevant knowledge already exists.

        Uses semantic similarity score if VectorStore is available.
        Otherwise uses keyword overlap heuristic.

        Args:
            query:     The query to check.
            threshold: Minimum similarity score to consider "known"
                       (0–1 scale, default 0.8).

        Returns:
            ``True`` if at least one entry meets the threshold.
        """
        if not self._entries:
            return False

        if self._ensure_vector_store() and self._vector_store.size > 0:
            docs = self._vector_store.query(query, top_k=1)
            if docs and docs[0].score >= threshold:
                logger.debug(
                    "has_knowledge=True (score=%.3f): %s",
                    docs[0].score, query,
                )
                return True
            return False

        # Fallback: exact substring match
        q_lower = query.strip().lower()
        for entry in self._entries.values():
            if q_lower in entry.query.lower() or q_lower in entry.content.lower():
                return True
        return False

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve an entry by ID."""
        return self._entries.get(entry_id)

    @property
    def size(self) -> int:
        return len(self._entries)

    # ── Backward-compatible helpers (used by WebSearch, AdminQuery) ──

    def lookup(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Legacy lookup (returns list of result dicts, or None).

        Used by the old WebSearch cache interface.
        """
        results = self.search(query, top_k=5)
        if not results:
            return None
        return [
            {
                "title": e.source.title() + " Knowledge",
                "snippet": e.content,
                "url": e.url or "",
            }
            for e in results
        ]

    def store_legacy(self, query: str, results: List[Dict[str, str]]) -> None:
        """Legacy store (accepts list of result dicts).

        Used by the old WebSearch / AdminQuery cache interface.
        """
        for r in results:
            source = r.get("source", "web")
            entry = KnowledgeEntry(
                query=query,
                content=r.get("snippet", r.get("body", "")),
                source=source,
                confidence=_DEFAULT_CONFIDENCE.get(source, 0.5),
                url=r.get("url", ""),
            )
            self.store(entry)

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                raw = json.loads(self.store_path.read_text(encoding="utf-8"))
                # Handle both new format (list of entries) and old format
                if isinstance(raw, list):
                    for d in raw:
                        entry = KnowledgeEntry.from_dict(d)
                        self._entries[entry.entry_id] = entry
                elif isinstance(raw, dict) and "entries" in raw:
                    for d in raw["entries"]:
                        entry = KnowledgeEntry.from_dict(d)
                        self._entries[entry.entry_id] = entry
                logger.info(
                    "Loaded %d knowledge entries from %s",
                    len(self._entries), self.store_path,
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load knowledge store: %s", exc)
                self._entries = {}
        else:
            self._entries = {}

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __repr__(self) -> str:
        return (
            f"KnowledgeStore(entries={self.size}, "
            f"vectors={'on' if self._use_vectors else 'off'}, "
            f"path={self.store_path})"
        )
