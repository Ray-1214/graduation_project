"""
KnowledgeStore — JSON-backed local cache for web search results.

Stores search results as ``{query → [results]}`` mappings, providing
a fast local lookup before hitting external APIs.  Results are indexed
by a normalised query key (lowered + stripped).

Persistence: auto-saves to ``rag/knowledge_base/search_cache.json``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "rag" / "knowledge_base" / "search_cache.json"
)


class KnowledgeStore:
    """Local cache for web search results.

    Args:
        cache_path: Path to the JSON cache file.
        max_age: Maximum age of a cached entry in seconds (default 24 h).
                 Entries older than this are treated as stale.
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        max_age: float = 86400.0,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE_PATH
        self.max_age = max_age
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ── Normalise query ──────────────────────────────────────────────

    @staticmethod
    def _key(query: str) -> str:
        return query.strip().lower()

    # ── Public API ───────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Return cached results for *query*, or ``None`` if absent/stale."""
        key = self._key(query)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.get("timestamp", 0) > self.max_age:
            logger.debug("Cache entry stale for '%s'", query)
            return None
        return entry["results"]

    def store(self, query: str, results: List[Dict[str, str]]) -> None:
        """Cache *results* under *query* and persist to disk."""
        key = self._key(query)
        self._cache[key] = {
            "query": query.strip(),
            "results": results,
            "timestamp": time.time(),
        }
        self._save()
        logger.info("Cached %d results for '%s'", len(results), query)

    @property
    def size(self) -> int:
        return len(self._cache)

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                logger.info("Loaded %d cached queries from %s", len(self._cache), self.cache_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load cache: %s", exc)
                self._cache = {}
        else:
            self._cache = {}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __repr__(self) -> str:
        return f"KnowledgeStore(entries={self.size}, path={self.cache_path})"
