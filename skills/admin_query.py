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
from typing import Optional

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
                  Only ``"sync"`` is implemented in Phase 1.
        """
        self._knowledge_store = knowledge_store
        self._store_initialised = knowledge_store is not None
        self.mode = mode

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

    # ── Mode B: Async placeholder (Phase 5) ──────────────────────────

    def _execute_async(self, question: str) -> str:
        """Write question to pending queue and return immediately."""
        # Phase 5: write to pending_queries.json or database
        # For now, fall back to sync mode with a warning
        logger.warning(
            "Async admin_query not yet implemented (Phase 5). "
            "Falling back to sync mode."
        )
        return self._execute_sync(question)

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
