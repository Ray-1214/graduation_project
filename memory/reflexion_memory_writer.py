"""
ReflexionMemoryWriter — routes structured reflexion insights to
the appropriate memory sub-systems.

Parses the three-section reflexion text produced by the
``strategy_reflection`` prompt template::

    【策略教訓】 → LongTermMemory   (source="reflexion")
    【知識收穫】 → KnowledgeStore   (source="reflexion", low confidence)
    【錯誤警告】 → SkillDocumentUpdater  (via existing update mechanism)
    【完整反思】 → kept in EpisodicLog  (backward-compatible)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from memory.episodic_log import EpisodicTrace
from memory.long_term import LongTermMemory, ReflectionEntry
from rag.knowledge_store import KnowledgeEntry, KnowledgeStore

if TYPE_CHECKING:
    from skill_graph.skill_document_updater import SkillDocumentUpdater
    from skill_graph.skill_node import SkillNode

logger = logging.getLogger(__name__)

# ── Confidence presets ───────────────────────────────────────────────
_CONFIDENCE_STRATEGY_SUCCESS = 0.7
_CONFIDENCE_STRATEGY_FAILURE = 0.5
_CONFIDENCE_KNOWLEDGE = 0.6
_KNOWLEDGE_DEDUP_THRESHOLD = 0.8


# ═══════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ReflexionInsight:
    """一條解析出的反思洞察。"""

    category: str           # "strategy" | "knowledge" | "warning"
    content: str            # 洞察內容
    source: str = "reflexion"
    confidence: float = 0.6
    episode_id: str = ""


@dataclass
class ReflexionCommitResult:
    """Summary of all memory writes performed by a single process() call."""

    strategy_lessons_written: int = 0
    knowledge_gains_written: int = 0
    knowledge_gains_deduplicated: int = 0
    warnings_dispatched: int = 0
    insights: List[ReflexionInsight] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
#  ReflexionMemoryWriter
# ═══════════════════════════════════════════════════════════════════════

class ReflexionMemoryWriter:
    """Routes structured reflexion output to the correct memory stores.

    Args:
        long_term:     LongTermMemory for strategy lessons.
        knowledge_store: KnowledgeStore for factual knowledge gains.
        doc_updater:   Optional SkillDocumentUpdater for error warnings.
    """

    def __init__(
        self,
        long_term: LongTermMemory,
        knowledge_store: KnowledgeStore,
        doc_updater: Optional["SkillDocumentUpdater"] = None,
    ) -> None:
        self.long_term = long_term
        self.knowledge_store = knowledge_store
        self.doc_updater = doc_updater

    # ── Public API ───────────────────────────────────────────────────

    def process(
        self,
        reflexion_text: str,
        trace: EpisodicTrace,
        success: bool,
        used_skills: Optional[List["SkillNode"]] = None,
    ) -> ReflexionCommitResult:
        """Parse reflexion text and route insights to memory stores.

        Args:
            reflexion_text: Agent's reflexion output (from
                ``strategy_reflection`` template).
            trace:         Execution trace for this episode.
            success:       Whether the task succeeded.
            used_skills:   Skills used during this episode.

        Returns:
            :class:`ReflexionCommitResult` summarising all writes.
        """
        result = ReflexionCommitResult()

        # 1. Parse sections
        sections = self._parse_sections(reflexion_text)

        strategy_lessons = sections.get("strategy", [])
        knowledge_gains = sections.get("knowledge", [])
        error_warnings = sections.get("warning", [])

        # Build insight list
        for s in strategy_lessons:
            conf = (
                _CONFIDENCE_STRATEGY_SUCCESS
                if success
                else _CONFIDENCE_STRATEGY_FAILURE
            )
            result.insights.append(ReflexionInsight(
                category="strategy",
                content=s,
                confidence=conf,
                episode_id=trace.task_id,
            ))
        for k in knowledge_gains:
            result.insights.append(ReflexionInsight(
                category="knowledge",
                content=k,
                confidence=_CONFIDENCE_KNOWLEDGE,
                episode_id=trace.task_id,
            ))
        for w in error_warnings:
            result.insights.append(ReflexionInsight(
                category="warning",
                content=w,
                episode_id=trace.task_id,
            ))

        # 2. Commit strategy lessons → LongTermMemory
        try:
            result.strategy_lessons_written = self._commit_strategy_lessons(
                strategy_lessons, trace, success,
            )
        except Exception as exc:
            msg = f"Failed to commit strategy lessons: {exc}"
            logger.error(msg)
            result.errors.append(msg)

        # 3. Commit knowledge gains → KnowledgeStore
        try:
            written, deduped = self._commit_knowledge_gains(
                knowledge_gains, trace,
            )
            result.knowledge_gains_written = written
            result.knowledge_gains_deduplicated = deduped
        except Exception as exc:
            msg = f"Failed to commit knowledge gains: {exc}"
            logger.error(msg)
            result.errors.append(msg)

        # 4. Dispatch error warnings → SkillDocumentUpdater
        try:
            result.warnings_dispatched = self._dispatch_warnings(
                error_warnings, trace, reflexion_text, success, used_skills,
            )
        except Exception as exc:
            msg = f"Failed to dispatch warnings: {exc}"
            logger.error(msg)
            result.errors.append(msg)

        logger.info(
            "ReflexionMemoryWriter: lessons=%d, knowledge=%d (dedup=%d), "
            "warnings=%d, errors=%d",
            result.strategy_lessons_written,
            result.knowledge_gains_written,
            result.knowledge_gains_deduplicated,
            result.warnings_dispatched,
            len(result.errors),
        )

        return result

    # ── Section parser ───────────────────────────────────────────────

    def _parse_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract 【策略教訓】, 【知識收穫】, 【錯誤警告】 sections.

        Falls back to treating the entire text as a single strategy
        lesson if no section markers are found (backward-compatible
        with old-format reflexion text).

        Returns:
            Dict with keys "strategy", "knowledge", "warning",
            each mapping to a list of extracted items.
        """
        sections: Dict[str, List[str]] = {
            "strategy": [],
            "knowledge": [],
            "warning": [],
        }

        # Find section boundaries
        markers = [
            ("strategy", r"【策略教訓】"),
            ("knowledge", r"【知識收穫】"),
            ("warning", r"【錯誤警告】"),
        ]

        found_any = False
        for key, pattern in markers:
            match = re.search(pattern, text)
            if match:
                found_any = True

        if not found_any:
            # Old-format reflexion: treat entire text as strategy lesson
            stripped = text.strip()
            if stripped:
                sections["strategy"] = [stripped]
            return sections

        # Extract each section's content
        for i, (key, pattern) in enumerate(markers):
            match = re.search(pattern, text)
            if not match:
                continue

            start = match.end()
            # Find end boundary: next section marker or end of text
            end = len(text)
            for j, (_, next_pattern) in enumerate(markers):
                if j <= i:
                    continue
                next_match = re.search(next_pattern, text[start:])
                if next_match:
                    end = start + next_match.start()
                    break

            section_text = text[start:end].strip()
            items = self._extract_items(section_text)
            sections[key] = items

        return sections

    @staticmethod
    def _extract_items(section_text: str) -> List[str]:
        """Extract individual items from a section block.

        Handles both bullet-point and line-by-line formats:
          - "- item text"
          - "1. item text"
          - "• item text"
          - Plain lines (one item per non-empty line)
        """
        items: List[str] = []
        for line in section_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading markers
            cleaned = re.sub(
                r"^(?:[-•]\s*|(?:\d+)[.、)]\s*)", "", line,
            ).strip()
            if cleaned and len(cleaned) > 3:
                items.append(cleaned)
        return items

    # ── Commit: strategy lessons → LongTermMemory ────────────────────

    def _commit_strategy_lessons(
        self,
        lessons: List[str],
        trace: EpisodicTrace,
        success: bool,
    ) -> int:
        """Write strategy lessons to LongTermMemory.

        Each lesson is stored as a :class:`ReflectionEntry` with
        ``source="reflexion"`` embedded in the format string.

        Returns:
            Number of entries written.
        """
        if not lessons:
            return 0

        confidence = (
            _CONFIDENCE_STRATEGY_SUCCESS
            if success
            else _CONFIDENCE_STRATEGY_FAILURE
        )

        task_type = trace.strategy or "unknown"
        written = 0

        for lesson_text in lessons:
            formatted = f"[Reflexion][{task_type}] {lesson_text}"

            entry = ReflectionEntry(
                task=trace.task_description,
                reflection=formatted,
                lessons=[lesson_text],
                score=confidence,
            )
            self.long_term.store(entry)
            written += 1
            logger.debug(
                "Strategy lesson → LTM: conf=%.2f episode=%s: %s",
                confidence, trace.task_id, lesson_text[:80],
            )

        return written

    # ── Commit: knowledge gains → KnowledgeStore ─────────────────────

    def _commit_knowledge_gains(
        self,
        gains: List[str],
        trace: EpisodicTrace,
    ) -> tuple[int, int]:
        """Write knowledge gains to KnowledgeStore.

        Uses deduplication: if the KnowledgeStore already has highly
        similar knowledge (> 0.8 threshold), skip that entry.

        Returns:
            (written_count, deduplicated_count)
        """
        if not gains:
            return 0, 0

        written = 0
        deduped = 0

        for gain_text in gains:
            # Deduplication check
            if self.knowledge_store.has_knowledge(
                gain_text, threshold=_KNOWLEDGE_DEDUP_THRESHOLD,
            ):
                logger.debug(
                    "Knowledge gain deduplicated: %s", gain_text[:80],
                )
                deduped += 1
                continue

            entry = KnowledgeEntry(
                query=f"[reflexion] {trace.task_description[:100]}",
                content=gain_text,
                source="reflexion",
                confidence=_CONFIDENCE_KNOWLEDGE,
            )
            self.knowledge_store.store(entry)
            written += 1
            logger.debug(
                "Knowledge gain → KS: episode=%s: %s",
                trace.task_id, gain_text[:80],
            )

        return written, deduped

    # ── Dispatch: error warnings → SkillDocumentUpdater ──────────────

    def _dispatch_warnings(
        self,
        warnings: List[str],
        trace: EpisodicTrace,
        reflexion_text: str,
        success: bool,
        used_skills: Optional[List["SkillNode"]],
    ) -> int:
        """Trigger SkillDocumentUpdater for each used skill.

        The warnings are already embedded in ``reflexion_text``
        which is passed through to the updater's existing
        ``update()`` method.

        Returns:
            Number of skills dispatched to.
        """
        if not warnings or not used_skills or self.doc_updater is None:
            return 0

        dispatched = 0
        for skill in used_skills:
            try:
                self.doc_updater.update(
                    skill=skill,
                    trace=trace,
                    reflexion_text=reflexion_text,
                    success=success,
                )
                dispatched += 1
                logger.debug(
                    "Warning dispatched to SkillDocumentUpdater: skill=%s",
                    skill.name,
                )
            except Exception as exc:
                logger.warning(
                    "SkillDocumentUpdater failed for skill '%s': %s",
                    skill.name, exc,
                )

        return dispatched

    def __repr__(self) -> str:
        return (
            f"ReflexionMemoryWriter("
            f"ltm={self.long_term!r}, "
            f"ks={self.knowledge_store!r}, "
            f"updater={'yes' if self.doc_updater else 'no'})"
        )
