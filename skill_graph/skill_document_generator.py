"""
SkillDocumentGenerator — creates structured Markdown skill manuals.

When the Evolution Operator Φ abstracts a new skill from reasoning traces,
this generator produces a human-readable (and LLM-consumable) Markdown
document describing how to use the skill.

Document structure
------------------
  # {skill_name}
  ## 適用場景           ← from initiation_set
  ## 前置條件           ← from dependency edges (if any)
  ## 背景知識           ← from KnowledgeStore entries (optional)
  ## 策略步驟           ← LLM-extracted or template-based
  ## 終止條件           ← from termination predicate
  ## 注意事項           ← initially empty, filled by Phase 3 reflection
  ## 版本歷史           ← auto-generated

Generation modes
----------------
  1. **LLM mode** (default): feeds traces to BaseLLM with a structured
     prompt, then validates the output.
  2. **Template fallback**: if no LLM is available, or the LLM output
     is too short / malformed, falls back to a deterministic template
     that directly inlines the trace steps.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from memory.episodic_log import EpisodicTrace, TraceStep
from skill_graph.skill_node import SkillNode

if TYPE_CHECKING:
    from core.llm_interface import BaseLLM
    from rag.knowledge_store import KnowledgeEntry

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
_DEFAULT_OUTPUT_DIR = "skills/learned"

# Minimum acceptable length (characters) for LLM-generated steps section
_MIN_LLM_OUTPUT_LEN = 80

# Pattern to detect tool actions in traces
_TOOL_ACTION_RE = re.compile(
    r"Action\[(\w+)\]\(([^)]*)\)",
)


# ═══════════════════════════════════════════════════════════════════
#  SkillDocumentGenerator
# ═══════════════════════════════════════════════════════════════════

class SkillDocumentGenerator:
    """Generates structured Markdown skill documents from traces.

    Args:
        llm:         Optional LLM backend for intelligent step
                     extraction.  Pass ``None`` to use template-only
                     mode.
        output_dir:  Directory for generated ``.md`` files, relative
                     to the project root.
        root:        Project root directory (absolute).  Defaults to
                     ``pathlib.Path.cwd()``.
    """

    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,
        output_dir: str = _DEFAULT_OUTPUT_DIR,
        root: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.output_dir = output_dir
        self.root = root or Path.cwd()

    # ── Public API ───────────────────────────────────────────────────

    def generate(
        self,
        skill: SkillNode,
        source_traces: List[EpisodicTrace],
        knowledge_entries: Optional[List["KnowledgeEntry"]] = None,
    ) -> str:
        """Generate a Markdown skill document and persist it.

        Args:
            skill:             The skill node to document.
            source_traces:     Traces from which the skill was abstracted.
            knowledge_entries: Optional related knowledge to include.

        Returns:
            The relative path to the generated ``.md`` file
            (also stored in ``skill.document_path``).
        """
        # Build each section
        sections: List[str] = []

        sections.append(f"# {skill.name}\n")

        sections.append(self._section_applicability(skill))
        sections.append(self._section_prerequisites(skill))

        # Optional: background knowledge
        if knowledge_entries:
            sections.append(self._section_knowledge(knowledge_entries))

        # Core: strategy steps (LLM-based or template)
        sections.append(
            self._section_strategy_steps(skill, source_traces)
        )

        sections.append(self._section_termination(skill))
        sections.append(self._section_caveats())
        sections.append(self._section_version_history(skill, source_traces))

        # Assemble
        content = "\n".join(sections)

        # Persist
        rel_path = self._write_document(skill.skill_id, content)
        skill.document_path = rel_path

        logger.info(
            "Generated skill document: %s (%d chars)",
            rel_path, len(content),
        )
        return rel_path

    # ── Section builders ─────────────────────────────────────────────

    def _section_applicability(self, skill: SkillNode) -> str:
        lines = ["## 適用場景\n"]
        if skill.initiation_set:
            for tag in skill.initiation_set:
                lines.append(f"- {tag}")
        else:
            lines.append("- （尚未定義適用場景）")
        lines.append("")
        return "\n".join(lines)

    def _section_prerequisites(self, skill: SkillNode) -> str:
        lines = ["## 前置條件\n"]
        # If parent_id exists, note the dependency
        if skill.parent_id:
            lines.append(f"- 繼承自技能 `{skill.parent_id}` (v{skill.version - 1})")
        else:
            lines.append("- 無特殊前置條件")
        lines.append("")
        return "\n".join(lines)

    def _section_knowledge(
        self,
        entries: List["KnowledgeEntry"],
    ) -> str:
        lines = ["## 背景知識\n"]
        for entry in entries[:5]:  # cap at 5
            source_label = {
                "web": "🌐 網路搜尋",
                "admin": "👤 管理員回覆",
                "rag": "📚 知識庫",
            }.get(entry.source, entry.source)
            lines.append(f"### {source_label}: {entry.query}\n")
            # Truncate very long content
            content = entry.content
            if len(content) > 300:
                content = content[:300] + "…"
            lines.append(content)
            if entry.url:
                lines.append(f"\n*來源: {entry.url}*")
            lines.append("")
        return "\n".join(lines)

    def _section_strategy_steps(
        self,
        skill: SkillNode,
        traces: List[EpisodicTrace],
    ) -> str:
        """Generate strategy steps — try LLM first, fall back to template."""
        # Try LLM generation
        if self.llm is not None:
            try:
                llm_output = self._generate_steps_with_llm(skill, traces)
                if llm_output and len(llm_output) >= _MIN_LLM_OUTPUT_LEN:
                    return f"## 策略步驟\n\n{llm_output}\n"
                else:
                    logger.warning(
                        "LLM output too short (%d chars), using fallback.",
                        len(llm_output) if llm_output else 0,
                    )
            except Exception as exc:
                logger.warning("LLM generation failed: %s, using fallback.", exc)

        # Fallback: template-based
        return self._template_strategy_steps(skill, traces)

    def _generate_steps_with_llm(
        self,
        skill: SkillNode,
        traces: List[EpisodicTrace],
    ) -> str:
        """Ask the LLM to distil trace steps into a strategy guide."""
        # Build trace summary for the prompt
        trace_text = self._format_traces_for_prompt(traces[:3])  # max 3

        prompt = (
            "你是一個技能文件生成器。根據以下推理軌跡，生成一份結構化的策略步驟指南。\n\n"
            f"技能名稱: {skill.name}\n"
            f"技能策略: {skill.policy}\n\n"
            f"推理軌跡:\n{trace_text}\n\n"
            "請用以下格式輸出操作步驟（繁體中文），只要步驟列表，不要前綴標題：\n"
            "1. 第一步...\n"
            "2. 第二步...\n"
            "（如果軌跡中有使用工具，請保留 Action[tool_name](\"參數\") 格式）\n\n"
            "步驟：\n"
        )

        result = self.llm.generate(
            prompt,
            max_tokens=512,
            temperature=0.3,
            stop=["##", "---"],
        )
        return result.strip()

    def _template_strategy_steps(
        self,
        skill: SkillNode,
        traces: List[EpisodicTrace],
    ) -> str:
        """Deterministic fallback: inline trace actions as steps."""
        lines = ["## 策略步驟\n"]

        if not traces:
            # No traces available — use policy directly
            lines.append("*（從策略描述生成）*\n")
            for i, step_text in enumerate(skill.policy.split(" → "), 1):
                lines.append(f"{i}. {step_text}")
            lines.append("")
            return "\n".join(lines)

        # Merge actions from the best trace (highest score)
        best_trace = max(traces, key=lambda t: t.score)
        lines.append(
            f"*（從 trace `{best_trace.task_id}` 提取, "
            f"score={best_trace.score:.2f}）*\n"
        )

        for i, step in enumerate(best_trace.steps, 1):
            action_text = step.action

            # Highlight tool actions
            tool_match = _TOOL_ACTION_RE.search(action_text)
            if tool_match:
                tool_name = tool_match.group(1)
                tool_arg = tool_match.group(2)
                lines.append(
                    f"{i}. 呼叫 `Action[{tool_name}](\"{tool_arg}\")` "
                    f"→ {step.outcome}"
                )
            else:
                lines.append(f"{i}. {action_text}")

                # Add outcome as sub-item if informative
                if step.outcome and step.outcome != action_text:
                    outcome_preview = step.outcome
                    if len(outcome_preview) > 120:
                        outcome_preview = outcome_preview[:120] + "…"
                    lines.append(f"   - 結果: {outcome_preview}")

        lines.append("")
        return "\n".join(lines)

    def _section_termination(self, skill: SkillNode) -> str:
        return (
            f"## 終止條件\n\n"
            f"{skill.termination}\n"
        )

    def _section_caveats(self) -> str:
        return (
            "## 注意事項\n\n"
            "*（尚無記錄 — Phase 3 反思機制將自動填入）*\n"
        )

    def _section_version_history(
        self,
        skill: SkillNode,
        traces: List[EpisodicTrace],
    ) -> str:
        lines = ["## 版本歷史\n"]
        date_str = datetime.fromtimestamp(skill.created_at).strftime(
            "%Y-%m-%d %H:%M"
        )
        trace_ids = ", ".join(t.task_id for t in traces[:3])
        lines.append(
            f"- v{skill.version} ({date_str}): "
            f"從 trace [{trace_ids}] 中首次學到"
        )
        if skill.parent_id:
            lines.append(
                f"- 演化自: `{skill.parent_id}`"
            )
        lines.append("")
        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────

    def _format_traces_for_prompt(
        self,
        traces: List[EpisodicTrace],
    ) -> str:
        """Format traces as a compact text block for LLM prompts."""
        parts: List[str] = []
        for t in traces:
            parts.append(f"--- Trace {t.task_id} (score={t.score:.2f}) ---")
            for j, step in enumerate(t.steps, 1):
                parts.append(
                    f"  Step {j}: [{step.action}] → {step.outcome}"
                )
        return "\n".join(parts)

    def _write_document(self, skill_id: str, content: str) -> str:
        """Write content to skills/learned/{skill_id}.md.

        Returns:
            Relative path from project root.
        """
        out_dir = self.root / self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{skill_id}.md"
        filepath = out_dir / filename
        filepath.write_text(content, encoding="utf-8")

        rel_path = f"{self.output_dir}/{filename}"
        return rel_path

    # ── Display ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        mode = "LLM" if self.llm else "template"
        return f"SkillDocumentGenerator(mode={mode}, dir={self.output_dir})"
