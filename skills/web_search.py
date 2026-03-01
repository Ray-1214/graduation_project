"""
Web search skill — real DuckDuckGo search with caching.

Uses the ``ddgs`` package.  Search results are cached in a
:class:`KnowledgeStore` so repeated or similar queries can
be served locally.

Features:
    • Top 3-5 results with title, snippet, URL
    • 10-second timeout per search
    • Rate limiting (≥ 1 s between requests)
    • Automatic KnowledgeStore write-through
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Dict, List, Optional

from skills.registry import BaseSkill

logger = logging.getLogger(__name__)

# Rate limiter — shared across threads
_last_search_time: float = 0.0
_rate_lock = threading.Lock()
_MIN_INTERVAL = 1.0  # seconds

_SEARCH_TIMEOUT = 10  # seconds
_MAX_RESULTS = 5


class WebSearch(BaseSkill):
    """Search the web via DuckDuckGo with local caching."""

    def __init__(self, knowledge_store=None) -> None:
        """
        Args:
            knowledge_store: Optional :class:`KnowledgeStore` instance.
                             If ``None``, a default one is created lazily.
        """
        self._knowledge_store = knowledge_store
        self._store_initialised = knowledge_store is not None

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information using DuckDuckGo. "
            "Input: a search query string. Returns top results with "
            "title, snippet, and URL."
        )

    # ── Lazy KnowledgeStore ──────────────────────────────────────────

    def _get_store(self):
        if not self._store_initialised:
            try:
                from rag.knowledge_store import KnowledgeStore
                self._knowledge_store = KnowledgeStore()
                self._store_initialised = True
            except Exception as exc:
                logger.warning("KnowledgeStore unavailable: %s", exc)
                self._store_initialised = True  # don't retry
        return self._knowledge_store

    # ── Rate limiting ────────────────────────────────────────────────

    @staticmethod
    def _wait_for_rate_limit() -> None:
        global _last_search_time
        with _rate_lock:
            now = time.time()
            elapsed = now - _last_search_time
            if elapsed < _MIN_INTERVAL:
                wait = _MIN_INTERVAL - elapsed
                logger.debug("Rate limit: waiting %.2fs", wait)
                time.sleep(wait)
            _last_search_time = time.time()

    # ── Core search ──────────────────────────────────────────────────

    def _search_ddg(self, query: str) -> List[Dict[str, str]]:
        """Execute a DuckDuckGo search with timeout.

        Returns a list of dicts: ``[{title, snippet, url}, ...]``
        """
        from ddgs import DDGS  # type: ignore

        self._wait_for_rate_limit()

        results: List[Dict[str, str]] = []
        try:
            ddgs = DDGS(timeout=_SEARCH_TIMEOUT)
            raw = ddgs.text(query, max_results=_MAX_RESULTS)
            for r in raw:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
        except Exception as exc:
            error_str = str(exc).lower()
            if "timeout" in error_str or "timed out" in error_str:
                raise TimeoutError(
                    f"Search timed out after {_SEARCH_TIMEOUT}s for query: {query}"
                ) from exc
            raise

        return results

    # ── Format output ────────────────────────────────────────────────

    @staticmethod
    def _format_results(query: str, results: List[Dict[str, str]], from_cache: bool = False) -> str:
        source = " (cached)" if from_cache else ""
        if not results:
            return f'[Search Results for "{query}"]{source}\nNo results found.'

        lines = [f'[Search Results for "{query}"]{source}']
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r['title']}")
            lines.append(f"   {r['snippet']}")
            lines.append(f"   URL: {r['url']}")
        return "\n".join(lines)

    # ── Execute (BaseSkill interface) ────────────────────────────────

    def execute(self, input_text: str) -> str:
        """Search the web for *input_text* and return formatted results.

        1. Check KnowledgeStore cache first.
        2. If miss, query DuckDuckGo.
        3. Cache results in KnowledgeStore.
        4. Return formatted string.
        """
        query = input_text.strip()
        if not query:
            return "Error: empty search query."

        # 1. Cache check
        store = self._get_store()
        if store is not None:
            cached = store.lookup(query)
            if cached is not None:
                logger.info("Cache hit for '%s' (%d results)", query, len(cached))
                return self._format_results(query, cached, from_cache=True)

        # 2. Live search
        try:
            results = self._search_ddg(query)
        except TimeoutError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            return f"Error: search failed — {exc}"

        # 3. Cache write-through
        if store is not None and results:
            try:
                store.store(query, results)
            except Exception as exc:
                logger.warning("Failed to cache results: %s", exc)

        # 4. Format and return
        return self._format_results(query, results)
