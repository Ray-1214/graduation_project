"""
AdminQuery skill — ask the human administrator a question.

Mode A (current): Synchronous CLI — prints the question to terminal
and blocks on ``input()`` until the admin types a response.

Mode B (Phase 5): Asynchronous API — writes to a pending queue and
returns immediately.  The ReAct loop resumes after the admin
responds via a REST endpoint or front-end UI.

Admin responses are automatically cached in :class:`KnowledgeStore`
with ``source="admin"`` for future retrieval.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from skills.registry import BaseSkill

logger = logging.getLogger(__name__)


class AdminQuery(BaseSkill):
    """Ask the administrator a question when other tools cannot help."""

    def __init__(self, knowledge_store=None, mode: str = "sync") -> None:
        """
        Args:
            knowledge_store: Optional :class:`KnowledgeStore` instance.
                             Created lazily if not provided.
            mode: ``"sync"`` (Mode A, CLI) or ``"async"`` (Mode B, API).
        """
        self._knowledge_store = knowledge_store
        self._store_initialised = knowledge_store is not None
        self.mode = mode

        # ── Async mode state (Mode B) ────────────────────────────────
        # Maps query_id → {event, question, response, episode_id, ...}
        self.pending_queries: Dict[str, Dict[str, Any]] = {}
        self._query_counter = 0

    @property
    def name(self) -> str:
        return "admin_query"

    @property
    def description(self) -> str:
        return (
            "Ask the admin a question when web search cannot provide "
            "the answer. Input: the question to ask."
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
                self._store_initialised = True
        return self._knowledge_store

    # ── Mode A: Synchronous (CLI) ────────────────────────────────────

    def _execute_sync(self, question: str) -> str:
        """Block on terminal input until the admin replies."""
        print("\n" + "=" * 60)
        print("  🔔 ADMIN QUERY — Agent needs your help")
        print("=" * 60)
        print(f"\n  Question: {question}\n")
        answer = input("  Your answer: ").strip()
        print("=" * 60 + "\n")

        if not answer:
            return "[Admin] (no response provided)"

        return f"[Admin] {answer}"

    # ── Mode B: Async API (Phase 5) ──────────────────────────────────

    def _execute_async(self, question: str) -> str:
        """Post question to pending queue and block until admin responds.

        The calling thread (agent executor) blocks on a
        ``threading.Event`` until :meth:`set_response` or
        :meth:`skip_query` is called from the API layer.
        """
        import threading
        import uuid

        query_id = f"q-{uuid.uuid4().hex[:12]}"
        event = threading.Event()

        self._query_counter += 1
        self.pending_queries[query_id] = {
            "query_id": query_id,
            "question": question,
            "event": event,
            "response": None,
            "skipped": False,
            "timestamp": time.time(),
            "context": "",
        }

        logger.info("Admin query posted: %s — %s", query_id, question)

        # Notify WebSocket listeners (callback set by backend)
        if hasattr(self, "_on_query_posted") and self._on_query_posted:
            try:
                self._on_query_posted(query_id, question)
            except Exception:
                pass

        # Block until admin responds or skips (timeout 10 min)
        event.wait(timeout=600)

        entry = self.pending_queries.pop(query_id, {})
        if entry.get("skipped"):
            return "[Admin] (question skipped — try another approach)"
        response = entry.get("response", "")
        if not response:
            return "[Admin] (no response — timed out after 10 minutes)"
        return f"[Admin] {response}"

    def set_response(self, query_id: str, response: str) -> bool:
        """Set admin response for a pending query (called by API)."""
        entry = self.pending_queries.get(query_id)
        if not entry:
            return False
        entry["response"] = response
        entry["event"].set()
        return True

    def skip_query(self, query_id: str) -> bool:
        """Skip a pending query (called by API)."""
        entry = self.pending_queries.get(query_id)
        if not entry:
            return False
        entry["skipped"] = True
        entry["event"].set()
        return True

    # ── BaseSkill interface ──────────────────────────────────────────

    def execute(self, input_text: str) -> str:
        """Ask the admin a question and return their response.

        The response is also cached in KnowledgeStore with
        ``source="admin"`` for future reference.
        """
        question = input_text.strip()
        if not question:
            return "Error: empty question."

        # Dispatch by mode
        if self.mode == "async":
            response = self._execute_async(question)
        else:
            response = self._execute_sync(question)

        # Cache the Q&A in KnowledgeStore
        store = self._get_store()
        if store is not None and not response.endswith("(no response provided)"):
            try:
                from rag.knowledge_store import KnowledgeEntry
                entry = KnowledgeEntry(
                    query=question,
                    content=response.removeprefix("[Admin] "),
                    source="admin",
                    confidence=0.9,
                )
                store.store(entry)
                logger.info("Admin response cached for: %s", question)
            except Exception as exc:
                logger.warning("Failed to cache admin response: %s", exc)

        return response
