"""
SkillDocumentUpdater — reflexion-driven updates to skill documents.

After an agent executes a task using a learned skill, the Reflexion
stage may produce insights:

  - **New strategies** (successful steps not in the .md)
  - **Failure warnings** (followed the .md but failed)
  - **Efficiency improvements** (shorter path to same result)

This module reads the existing ``.md`` skill document, analyses the
new trace + reflexion text, and patches the document in-place —
incrementing the version and appending to the version history.

Trigger conditions
~~~~~~~~~~~~~~~~~~
Not every task execution triggers an update.  Updates are triggered when:

  1. **Task failure** — always add to the "注意事項" (caveats) section.
  2. **Success + new discovery** — reflexion text contains new insights.
  3. **Significant divergence** — trace steps differ markedly from the
     documented strategy steps.

Safety
~~~~~~
- Old document is backed up as ``{skill_id}_v{old_ver}.md.bak``
  before overwriting.
- If the LLM-generated update is too short or malformed, the original
  document is preserved unchanged.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from memory.episodic_log import EpisodicTrace
from skill_graph.skill_node import SkillNode

if TYPE_CHECKING:
    from core.llm_interface import BaseLLM

logger = logging.getLogger(__name__)

# ── Trigger keywords that suggest new discoveries ────────────────────
_DISCOVERY_KEYWORDS = [
    "發現", "新方法", "更好", "改進", "替代", "意外", "不同",
    "原來", "其實", "可以改", "更快", "更有效",
    "discover", "found", "better", "improve", "alternative",
    "unexpected", "instead", "optimiz",
]

# Minimum quality bar for LLM-generated updates
_MIN_UPDATE_LENGTH = 100


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class UpdateResult:
    """Outcome of an update attempt.

    Attributes:
        updated:        Whether the document was actually modified.
        trigger:        Which trigger condition fired (or None).
        backup_path:    Path to the backup file (or None).
        new_version:    Version number after update (or current).
        reason:         Human-readable explanation of what happened.
    """
    updated: bool = False
    trigger: Optional[str] = None
    backup_path: Optional[str] = None
    new_version: int = 1
    reason: str = ""


# ═══════════════════════════════════════════════════════════════════
#  SkillDocumentUpdater
# ═══════════════════════════════════════════════════════════════════

class SkillDocumentUpdater:
    """Updates existing skill ``.md`` documents based on reflexion.

    Args:
        llm:                Optional LLM backend for intelligent edits.
                            Pass ``None`` for template-only updates.
        root:               Project root directory.
        divergence_threshold:  Minimum divergence ratio (0–1) between
                            trace actions and documented steps to trigger
                            an update on success.
    """

    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,
        root: Optional[Path] = None,
        divergence_threshold: float = 0.4,
    ) -> None:
        self.llm = llm
        self.root = root or Path.cwd()
        self.divergence_threshold = divergence_threshold

    # ── Public API ───────────────────────────────────────────────────

    def update(
        self,
        skill: SkillNode,
        trace: EpisodicTrace,
        reflexion_text: str,
        success: bool,
    ) -> UpdateResult:
        """Conditionally update a skill document after execution.

        Args:
            skill:           The skill that was used.
            trace:           Execution trace from this episode.
            reflexion_text:  Agent's reflexion output.
            success:         Whether the task succeeded.

        Returns:
            :class:`UpdateResult` describing what happened.
        """
        # ── Guard: no document to update ──────────────────────────
        if not skill.document_path:
            return UpdateResult(reason="No document_path set on skill")

        doc_path = self.root / skill.document_path
        if not doc_path.exists():
            return UpdateResult(reason=f"Document not found: {doc_path}")

        original_content = doc_path.read_text(encoding="utf-8")

        # ── Check trigger conditions ─────────────────────────────
        trigger = self._check_triggers(
            skill, trace, reflexion_text, success, original_content,
        )
        if trigger is None:
            return UpdateResult(
                reason="No trigger condition met",
                new_version=skill.version,
            )

        # ── Generate updated content ─────────────────────────────
        new_content = self._generate_update(
            skill, trace, reflexion_text, success, trigger,
            original_content,
        )

        # ── Quality gate ─────────────────────────────────────────
        if not self._passes_quality_gate(new_content, original_content):
            return UpdateResult(
                trigger=trigger,
                reason="LLM update failed quality gate, original preserved",
                new_version=skill.version,
            )

        # ── Backup old version ───────────────────────────────────
        backup_path = self._backup(skill, doc_path)

        # ── Write new version ────────────────────────────────────
        doc_path.write_text(new_content, encoding="utf-8")

        # ── Update skill metadata ────────────────────────────────
        old_version = skill.version
        skill.version += 1
        skill.updated_at = time.time()

        logger.info(
            "Updated skill doc '%s' v%d→v%d (trigger=%s)",
            skill.skill_id, old_version, skill.version, trigger,
        )

        return UpdateResult(
            updated=True,
            trigger=trigger,
            backup_path=str(backup_path),
            new_version=skill.version,
            reason=f"Document updated: {trigger}",
        )

    # ── Trigger detection ────────────────────────────────────────────

    def _check_triggers(
        self,
        skill: SkillNode,
        trace: EpisodicTrace,
        reflexion_text: str,
        success: bool,
        document_content: str,
    ) -> Optional[str]:
        """Determine which trigger condition fires (if any).

        Returns:
            Trigger name string, or None if no trigger.
        """
        # T1: Task failure → always update caveats
        if not success:
            return "failure_caveat"

        # T2: Success + reflexion contains discovery keywords
        if self._has_discovery(reflexion_text):
            return "new_discovery"

        # T3: Significant divergence between trace and document
        if self._has_divergence(trace, document_content):
            return "strategy_divergence"

        return None

    def _has_discovery(self, reflexion_text: str) -> bool:
        """Check if reflexion text mentions new discoveries."""
        text_lower = reflexion_text.lower()
        return any(kw.lower() in text_lower for kw in _DISCOVERY_KEYWORDS)

    def _has_divergence(
        self,
        trace: EpisodicTrace,
        document_content: str,
    ) -> bool:
        """Check if trace actions diverge significantly from document."""
        trace_actions = " ".join(step.action for step in trace.steps)
        # Extract strategy steps section from the document
        strategy_section = self._extract_section(
            document_content, "策略步驟",
        )
        if not strategy_section:
            return False

        sim = SequenceMatcher(
            None, trace_actions.lower(), strategy_section.lower(),
        ).ratio()
        # Low similarity = high divergence
        return (1.0 - sim) >= self.divergence_threshold

    # ── Update generation ────────────────────────────────────────────

    def _generate_update(
        self,
        skill: SkillNode,
        trace: EpisodicTrace,
        reflexion_text: str,
        success: bool,
        trigger: str,
        original_content: str,
    ) -> str:
        """Generate the updated document content."""
        if self.llm is not None:
            try:
                result = self._llm_update(
                    skill, trace, reflexion_text, success, trigger,
                    original_content,
                )
                if result and len(result.strip()) >= _MIN_UPDATE_LENGTH:
                    return result
                logger.warning(
                    "LLM update too short (%d chars), using template",
                    len(result.strip()) if result else 0,
                )
            except Exception as exc:
                logger.warning("LLM update failed: %s, using template", exc)

        # Template fallback
        return self._template_update(
            skill, trace, reflexion_text, success, trigger,
            original_content,
        )

    def _llm_update(
        self,
        skill: SkillNode,
        trace: EpisodicTrace,
        reflexion_text: str,
        success: bool,
        trigger: str,
        original_content: str,
    ) -> str:
        """Use LLM to produce an updated document."""
        trace_text = "\n".join(
            f"  {i+1}. [{s.action}] → {s.outcome}"
            for i, s in enumerate(trace.steps)
        )

        prompt = (
            f"你是一個技能文件編輯器。以下是 skill「{skill.name}」的現有文件：\n\n"
            f"--- 現有文件 ---\n{original_content}\n--- 現有文件結束 ---\n\n"
            f"本次執行 trace：\n{trace_text}\n\n"
            f"反思：{reflexion_text}\n\n"
            f"任務結果：{'成功' if success else '失敗'}\n"
            f"觸發原因：{trigger}\n\n"
            f"請根據以上資訊，生成更新後的完整 Markdown 文件。規則：\n"
        )

        if trigger == "failure_caveat":
            prompt += (
                "- 在「## 注意事項」段落中追加本次失敗的教訓\n"
                "- 保持其他段落不變\n"
            )
        elif trigger == "new_discovery":
            prompt += (
                "- 在「## 策略步驟」中補充新發現的有效策略\n"
                "- 如果有新的適用場景，更新「## 適用場景」\n"
            )
        elif trigger == "strategy_divergence":
            prompt += (
                "- 根據 trace 中實際有效的步驟，更新「## 策略步驟」\n"
                "- 保留原有步驟中仍然有效的部分\n"
            )

        prompt += (
            f"\n- 在「## 版本歷史」中追加 v{skill.version + 1} 記錄\n"
            f"- 輸出完整的 Markdown 文件（不要省略任何段落）\n"
        )

        return self.llm.generate(prompt)

    def _template_update(
        self,
        skill: SkillNode,
        trace: EpisodicTrace,
        reflexion_text: str,
        success: bool,
        trigger: str,
        original_content: str,
    ) -> str:
        """Deterministic template-based update."""
        content = original_content
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_ver = skill.version + 1
        trace_id = trace.task_id

        if trigger == "failure_caveat":
            content = self._append_to_section(
                content, "注意事項",
                f"\n- ⚠️ [{now_str}] 任務失敗 (trace: {trace_id}): "
                f"{reflexion_text[:200]}",
            )
        elif trigger == "new_discovery":
            content = self._append_to_section(
                content, "策略步驟",
                f"\n\n**[v{new_ver} 新增]** 根據 trace {trace_id} 的反思：\n"
                f"{reflexion_text[:300]}",
            )
        elif trigger == "strategy_divergence":
            # Append a note about the divergence
            trace_summary = " → ".join(s.action for s in trace.steps[:5])
            content = self._append_to_section(
                content, "策略步驟",
                f"\n\n**[v{new_ver} 替代路徑]** trace {trace_id} "
                f"使用了不同步驟：{trace_summary}",
            )

        # Append version history entry
        trace_ids = trace_id
        content = self._append_to_section(
            content, "版本歷史",
            f"\n| v{new_ver} | {now_str} | "
            f"{trigger} — {trace_ids} |",
        )

        return content

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_section(content: str, section_name: str) -> str:
        """Extract text under a ## heading until the next ## or EOF."""
        pattern = rf"## {re.escape(section_name)}\s*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _append_to_section(
        content: str,
        section_name: str,
        text_to_append: str,
    ) -> str:
        """Append text at the end of a named section."""
        pattern = rf"(## {re.escape(section_name)}\s*\n.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            insertion_point = match.end()
            return (
                content[:insertion_point]
                + text_to_append
                + content[insertion_point:]
            )
        # Section not found — append at end
        return content + f"\n\n## {section_name}\n{text_to_append}\n"

    def _backup(self, skill: SkillNode, doc_path: Path) -> Path:
        """Create a backup of the current document."""
        backup_name = f"{skill.skill_id}_v{skill.version}.md.bak"
        backup_path = doc_path.parent / backup_name
        shutil.copy2(doc_path, backup_path)
        return backup_path

    @staticmethod
    def _passes_quality_gate(
        new_content: str,
        original_content: str,
    ) -> bool:
        """Check if the updated content meets quality standards."""
        if not new_content or not new_content.strip():
            return False
        # Must be at least as long as 60% of original
        if len(new_content.strip()) < len(original_content.strip()) * 0.6:
            return False
        # Must still contain key section headers
        required = ["#", "策略步驟"]
        return all(kw in new_content for kw in required)

    # ── Display ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        mode = "LLM" if self.llm else "template"
        return (
            f"SkillDocumentUpdater(mode={mode}, "
            f"divergence_threshold={self.divergence_threshold})"
        )
