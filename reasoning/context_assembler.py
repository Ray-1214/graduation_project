"""
ContextAssembler — Slot-based context window management.

Implements the "Context Engineering" pattern (Karpathy / Tobi Lutke, 2025):
every token in the context window is precious — assemble, budget, and
compress systematically rather than naïvely concatenating.

Designed for Mistral-7B's 4096-token context window but works with any
max_total_tokens setting.

Key concepts:
  - **ContextSlot**: a named, prioritised region of the prompt.
  - **Slot budgets**: hard token caps per slot, with a total budget.
  - **Compression cascade**: when the total exceeds budget, compressible
    slots are summarised in ascending priority order (highest priority
    number = least important = compressed first).
  - **Recency-weighted history**: recent ReAct steps kept verbatim;
    older steps compressed to one-line summaries.
  - **Knowledge relevance scoring**: KnowledgeStore entries ranked by
    similarity, reflexion-sourced knowledge separated from web/admin.

References:
  - Anthropic, "Building Effective Agents" (2024)
  - Simon Willison, "Context Engineering" (2025)
  - Karpathy, "Context Engineering" tweet thread (2025)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from core.llm_interface import BaseLLM

logger = logging.getLogger(__name__)

# ── Token estimation ─────────────────────────────────────────────────
# Mistral tokeniser isn't available in this environment.
# Use character-based heuristic: ~4 chars per token for mixed CJK/EN.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ═══════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ContextSlot:
    """A named, prioritised region of the context window.

    Attributes:
        name:           Slot identifier (e.g. "system", "task").
        priority:       0 = highest (never compressed), 9 = lowest.
        max_tokens:     Token budget for this slot.
        content:        Actual text content.
        compressible:   Whether this slot can be summarised.
        original_tokens: Token count before compression (for reporting).
        compressed:     Whether compression was applied.
    """

    name: str
    priority: int = 5
    max_tokens: int = 500
    content: str = ""
    compressible: bool = True
    original_tokens: int = 0
    compressed: bool = False


@dataclass
class ReActStep:
    """A single step from a ReAct loop for history assembly.

    Attributes:
        thought:     The reasoning text.
        action:      Tool name + input.
        observation: Tool output.
        step_num:    Step index (1-based).
    """

    thought: str = ""
    action: str = ""
    observation: str = ""
    step_num: int = 0


@dataclass
class AssembledContext:
    """Result of :meth:`ContextAssembler.assemble`.

    Attributes:
        prompt:         The final assembled prompt string.
        total_tokens:   Estimated total token count.
        slot_report:    Per-slot usage {name: {budget, used, compressed}}.
        overflow:       True if compression was needed.
        compressed_slots: Names of slots that were compressed.
    """

    prompt: str = ""
    total_tokens: int = 0
    slot_report: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overflow: bool = False
    compressed_slots: List[str] = field(default_factory=list)


# ── Default slot budgets ─────────────────────────────────────────────

DEFAULT_SLOT_BUDGETS: Dict[str, Dict[str, Any]] = {
    "system":    {"priority": 0, "max_tokens": 300,  "compressible": False},
    "task":      {"priority": 0, "max_tokens": 200,  "compressible": False},
    "knowledge": {"priority": 2, "max_tokens": 500,  "compressible": True},
    "lessons":   {"priority": 3, "max_tokens": 400,  "compressible": True},
    "skills":    {"priority": 4, "max_tokens": 300,  "compressible": True},
    "history":   {"priority": 5, "max_tokens": 1500, "compressible": True},
    "tools":     {"priority": 1, "max_tokens": 300,  "compressible": False},
    "reserve":   {"priority": 9, "max_tokens": 500,  "compressible": False},
}


# ═══════════════════════════════════════════════════════════════════════
#  ContextAssembler
# ═══════════════════════════════════════════════════════════════════════

class ContextAssembler:
    """Slot-based context window assembler.

    Args:
        max_total_tokens: Overall token budget for the context window.
        slot_budgets:     Custom slot budget overrides (merged with defaults).
        llm:              Optional LLM for compression summaries.
    """

    def __init__(
        self,
        max_total_tokens: int = 4000,
        slot_budgets: Optional[Dict[str, Dict[str, Any]]] = None,
        llm: Optional["BaseLLM"] = None,
    ) -> None:
        self.max_total_tokens = max_total_tokens
        self.llm = llm

        # Merge user budgets with defaults
        self._budgets: Dict[str, Dict[str, Any]] = {
            **DEFAULT_SLOT_BUDGETS,
        }
        if slot_budgets:
            self._budgets.update(slot_budgets)

        # Last assembly report
        self._last_report: Dict[str, Dict[str, Any]] = {}

    # ── Public API ───────────────────────────────────────────────────

    def assemble(
        self,
        task: str,
        system_prompt: str = "",
        history: Optional[List[ReActStep]] = None,
        knowledge: Optional[List[Any]] = None,
        lessons: Optional[List[str]] = None,
        skill_docs: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        *,
        # Legacy kwargs for backward compatibility
        skills_block: Optional[str] = None,
        rag_context: Optional[str] = None,
        thinking_plan: Optional[str] = None,
    ) -> AssembledContext:
        """Assemble the full context from all sources.

        Args:
            task:          The task description.
            system_prompt: System-level instructions.
            history:       ReAct step history.
            knowledge:     KnowledgeEntry objects from KnowledgeStore.
            lessons:       Reflexion lesson strings.
            skill_docs:    Skill documentation blocks.
            tools:         Tool description strings.
            skills_block:  (legacy) Pre-formatted skill block.
            rag_context:   (legacy) Pre-formatted RAG context.
            thinking_plan: (legacy) Phase ① plan text.

        Returns:
            :class:`AssembledContext` with prompt and budget report.
        """
        # Build raw slots
        slots = self._build_slots(
            task=task,
            system_prompt=system_prompt,
            history=history or [],
            knowledge=knowledge or [],
            lessons=lessons or [],
            skill_docs=skill_docs or [],
            tools=tools or [],
            skills_block=skills_block,
            rag_context=rag_context,
            thinking_plan=thinking_plan,
        )

        # Enforce per-slot limits
        for slot in slots:
            slot.original_tokens = estimate_tokens(slot.content)
            if slot.original_tokens > slot.max_tokens:
                if slot.compressible:
                    slot = self._compress_slot(slot)
                else:
                    # Hard truncate non-compressible slots
                    max_chars = slot.max_tokens * _CHARS_PER_TOKEN
                    slot.content = slot.content[:max_chars]

        # Check total
        total = sum(estimate_tokens(s.content) for s in slots)
        overflow = total > self.max_total_tokens
        compressed_names: List[str] = []

        if overflow:
            # Compression cascade: compress from lowest priority first
            compressible = sorted(
                [s for s in slots if s.compressible and s.content],
                key=lambda s: -s.priority,  # highest number first
            )
            for slot in compressible:
                if total <= self.max_total_tokens:
                    break
                old_tokens = estimate_tokens(slot.content)
                # Target: reduce to 60% of current or max_tokens, whichever smaller
                target = min(
                    int(old_tokens * 0.6),
                    slot.max_tokens,
                )
                slot = self._compress_slot(slot, target_tokens=target)
                new_tokens = estimate_tokens(slot.content)
                total -= (old_tokens - new_tokens)
                compressed_names.append(slot.name)

        # Build final prompt from slots (ordered by priority, then position)
        ordered = sorted(slots, key=lambda s: s.priority)
        prompt_parts: List[str] = []
        for slot in ordered:
            if slot.content.strip() and slot.name != "reserve":
                prompt_parts.append(slot.content)

        prompt = "\n\n".join(prompt_parts)
        final_tokens = estimate_tokens(prompt)

        # Build report
        report: Dict[str, Dict[str, Any]] = {}
        for slot in slots:
            report[slot.name] = {
                "budget": slot.max_tokens,
                "original": slot.original_tokens,
                "used": estimate_tokens(slot.content),
                "compressed": slot.compressed,
            }
        self._last_report = report

        result = AssembledContext(
            prompt=prompt,
            total_tokens=final_tokens,
            slot_report=report,
            overflow=overflow,
            compressed_slots=compressed_names,
        )

        logger.info(
            "Context assembled: %d tokens (budget=%d), overflow=%s, "
            "compressed=%s",
            final_tokens, self.max_total_tokens, overflow,
            compressed_names or "none",
        )

        return result

    def get_budget_report(self) -> Dict[str, Dict[str, Any]]:
        """Return the slot usage report from the last assembly."""
        return dict(self._last_report)

    # ── Slot construction ────────────────────────────────────────────

    def _build_slots(
        self,
        task: str,
        system_prompt: str,
        history: List[ReActStep],
        knowledge: List[Any],
        lessons: List[str],
        skill_docs: List[str],
        tools: List[str],
        skills_block: Optional[str],
        rag_context: Optional[str],
        thinking_plan: Optional[str],
    ) -> List[ContextSlot]:
        """Create ContextSlot objects from raw inputs."""
        slots: List[ContextSlot] = []

        # System slot
        sys_cfg = self._budgets["system"]
        slots.append(ContextSlot(
            name="system",
            priority=sys_cfg["priority"],
            max_tokens=sys_cfg["max_tokens"],
            content=system_prompt or "",
            compressible=sys_cfg["compressible"],
        ))

        # Task slot
        task_cfg = self._budgets["task"]
        task_content = task
        if thinking_plan:
            task_content = f"{task}\n\n[任務分析計劃]\n{thinking_plan}\n[計劃結束]"
        slots.append(ContextSlot(
            name="task",
            priority=task_cfg["priority"],
            max_tokens=task_cfg["max_tokens"],
            content=task_content,
            compressible=task_cfg["compressible"],
        ))

        # Knowledge slot
        k_cfg = self._budgets["knowledge"]
        knowledge_text = self._format_knowledge(
            knowledge, rag_context, k_cfg["max_tokens"],
        )
        slots.append(ContextSlot(
            name="knowledge",
            priority=k_cfg["priority"],
            max_tokens=k_cfg["max_tokens"],
            content=knowledge_text,
            compressible=k_cfg["compressible"],
        ))

        # Lessons slot
        l_cfg = self._budgets["lessons"]
        lessons_text = self._format_lessons(lessons)
        slots.append(ContextSlot(
            name="lessons",
            priority=l_cfg["priority"],
            max_tokens=l_cfg["max_tokens"],
            content=lessons_text,
            compressible=l_cfg["compressible"],
        ))

        # Skills slot
        s_cfg = self._budgets["skills"]
        skills_text = self._format_skills(skill_docs, skills_block)
        slots.append(ContextSlot(
            name="skills",
            priority=s_cfg["priority"],
            max_tokens=s_cfg["max_tokens"],
            content=skills_text,
            compressible=s_cfg["compressible"],
        ))

        # History slot (recency-weighted)
        h_cfg = self._budgets["history"]
        history_text = self._compress_history(history, h_cfg["max_tokens"])
        slots.append(ContextSlot(
            name="history",
            priority=h_cfg["priority"],
            max_tokens=h_cfg["max_tokens"],
            content=history_text,
            compressible=h_cfg["compressible"],
        ))

        # Tools slot
        t_cfg = self._budgets["tools"]
        tools_text = "\n".join(tools) if tools else ""
        slots.append(ContextSlot(
            name="tools",
            priority=t_cfg["priority"],
            max_tokens=t_cfg["max_tokens"],
            content=tools_text,
            compressible=t_cfg["compressible"],
        ))

        # Reserve slot (empty — just budget placeholder for LLM output)
        r_cfg = self._budgets["reserve"]
        slots.append(ContextSlot(
            name="reserve",
            priority=r_cfg["priority"],
            max_tokens=r_cfg["max_tokens"],
            content="",
            compressible=r_cfg["compressible"],
        ))

        return slots

    # ── Formatting helpers ───────────────────────────────────────────

    def _format_knowledge(
        self,
        entries: List[Any],
        rag_context: Optional[str],
        max_tokens: int,
    ) -> str:
        """Format knowledge entries with relevance scoring.

        Strategy:
          1. Separate reflexion-sourced and external-sourced entries.
          2. Sort each group by confidence (descending).
          3. Fill up to token budget, prioritising external sources.
        """
        if not entries and not rag_context:
            return ""

        # If using legacy rag_context string, return it directly
        if not entries and rag_context:
            return f"[已知知識]\n{rag_context}"

        # Split by source
        external = []
        reflexion = []
        for entry in entries:
            src = getattr(entry, "source", "web")
            conf = getattr(entry, "confidence", 0.5)
            content = getattr(entry, "content", str(entry))
            query = getattr(entry, "query", "")

            item = {"content": content, "confidence": conf, "query": query,
                    "source": src}

            if src == "reflexion":
                reflexion.append(item)
            else:
                external.append(item)

        # Sort by confidence descending
        external.sort(key=lambda x: -x["confidence"])
        reflexion.sort(key=lambda x: -x["confidence"])

        # Build text within budget
        parts: List[str] = []
        tokens_used = 0

        if external:
            parts.append("[已知知識（外部來源）]")
            for item in external:
                line = f"- [{item['source']}][conf={item['confidence']:.1f}] {item['content']}"
                line_tokens = estimate_tokens(line)
                if tokens_used + line_tokens > max_tokens:
                    break
                parts.append(line)
                tokens_used += line_tokens

        if reflexion:
            parts.append("[Agent 經驗（reflexion，可信度較低）]")
            for item in reflexion:
                line = f"- [reflexion][conf={item['confidence']:.1f}] {item['content']}"
                line_tokens = estimate_tokens(line)
                if tokens_used + line_tokens > max_tokens:
                    break
                parts.append(line)
                tokens_used += line_tokens

        if rag_context:
            rag_tokens = estimate_tokens(rag_context)
            if tokens_used + rag_tokens <= max_tokens:
                parts.append(f"[RAG 補充]\n{rag_context}")

        return "\n".join(parts)

    @staticmethod
    def _format_lessons(lessons: List[str]) -> str:
        """Format reflexion lessons."""
        if not lessons:
            return ""
        header = "[歷史教訓]"
        items = [f"- {lesson}" for lesson in lessons]
        return header + "\n" + "\n".join(items)

    @staticmethod
    def _format_skills(
        skill_docs: List[str],
        skills_block: Optional[str],
    ) -> str:
        """Format skill documentation."""
        if skills_block:
            return skills_block
        if not skill_docs:
            return ""
        return "[可用技能]\n" + "\n---\n".join(skill_docs)

    # ── Recency-Weighted History Compression ─────────────────────────

    def _compress_history(
        self,
        steps: List[ReActStep],
        max_tokens: int,
    ) -> str:
        """Build recency-weighted history text.

        Tiers:
          - Last 2 steps: full verbatim (thought + action + observation).
          - Steps 3-5:    thought + action summary (observation truncated).
          - Steps 6+:     single-line aggregate summary.

        Args:
            steps:      Ordered list of ReAct steps.
            max_tokens: Token budget for the history slot.

        Returns:
            Formatted history string.
        """
        if not steps:
            return ""

        n = len(steps)
        parts: List[str] = []

        # Tier 3: very old steps (6+) → one-line summary
        if n > 5:
            old_steps = steps[:n - 5]
            summary = self._summarise_old_steps(old_steps)
            parts.append(summary)

        # Tier 2: middle steps (3-5 from end) → condensed
        mid_start = max(0, n - 5)
        mid_end = max(0, n - 2)
        for step in steps[mid_start:mid_end]:
            condensed = self._condense_step(step)
            parts.append(condensed)

        # Tier 1: last 2 steps → full verbatim
        recent_start = max(0, n - 2)
        for step in steps[recent_start:]:
            full = self._format_full_step(step)
            parts.append(full)

        history_text = "\n".join(parts)

        # Post-trim if still over budget
        while (estimate_tokens(history_text) > max_tokens
               and len(parts) > 2):
            # Remove oldest part
            parts.pop(0)
            history_text = "\n".join(parts)

        return history_text

    @staticmethod
    def _format_full_step(step: ReActStep) -> str:
        """Format a ReAct step fully."""
        lines = []
        if step.thought:
            lines.append(f"Thought: {step.thought}")
        if step.action:
            lines.append(f"Action: {step.action}")
        if step.observation:
            lines.append(f"Observation: {step.observation}")
        return "\n".join(lines)

    @staticmethod
    def _condense_step(step: ReActStep) -> str:
        """Format a ReAct step in condensed form."""
        parts = []
        if step.thought:
            # Keep first 80 chars of thought
            parts.append(f"Thought: {step.thought[:80]}…")
        if step.action:
            parts.append(f"Action: {step.action}")
        if step.observation:
            # Truncate observation aggressively
            parts.append(f"Obs: {step.observation[:50]}…")
        return " | ".join(parts)

    @staticmethod
    def _summarise_old_steps(steps: List[ReActStep]) -> str:
        """Create a one-line summary of old steps."""
        if not steps:
            return ""
        actions = []
        for s in steps:
            if s.action:
                actions.append(s.action.split(":")[0].strip())
        n = len(steps)
        if actions:
            unique_actions = list(dict.fromkeys(actions))  # deduplicate, preserve order
            return f"[步驟 1-{n} 摘要: 執行了 {', '.join(unique_actions[:5])}]"
        return f"[步驟 1-{n}: {n} 個推理步驟]"

    # ── Slot compression ─────────────────────────────────────────────

    def _compress_slot(
        self,
        slot: ContextSlot,
        target_tokens: Optional[int] = None,
    ) -> ContextSlot:
        """Compress a single slot to fit its budget.

        Uses LLM summarisation if available, otherwise truncates
        intelligently.

        Args:
            slot:          The slot to compress.
            target_tokens: Target token count after compression.
                           Defaults to slot.max_tokens.

        Returns:
            The same slot object (mutated) with compressed content.
        """
        if not slot.content:
            return slot

        target = target_tokens or slot.max_tokens
        current = estimate_tokens(slot.content)

        if current <= target:
            return slot

        slot.original_tokens = current

        # Try LLM summarisation
        if self.llm is not None:
            try:
                compressed = self._llm_compress(slot.content, target)
                slot.content = compressed
                slot.compressed = True
                return slot
            except Exception as exc:
                logger.warning("LLM compression failed for '%s': %s",
                               slot.name, exc)

        # Fallback: intelligent truncation
        slot.content = self._truncate(slot.content, target)
        slot.compressed = True
        return slot

    def _llm_compress(self, text: str, target_tokens: int) -> str:
        """Use LLM to summarise text to target length."""
        target_chars = target_tokens * _CHARS_PER_TOKEN
        prompt = (
            f"請將以下文字壓縮為約 {target_chars} 字的摘要，"
            f"保留最重要的資訊：\n\n{text}"
        )
        result = self.llm.generate(prompt, max_tokens=target_tokens)
        return result.strip()

    @staticmethod
    def _truncate(text: str, target_tokens: int) -> str:
        """Intelligently truncate text to target token count.

        Preserves complete lines where possible, keeping the most
        recent content (end of text).
        """
        target_chars = target_tokens * _CHARS_PER_TOKEN
        if len(text) <= target_chars:
            return text

        # Split into lines and keep from the end
        lines = text.split("\n")
        kept: List[str] = []
        chars = 0
        for line in reversed(lines):
            if chars + len(line) + 1 > target_chars:
                break
            kept.insert(0, line)
            chars += len(line) + 1

        if kept:
            return "[...壓縮...]\n" + "\n".join(kept)
        # If even one line is too long, hard truncate
        return "[...壓縮...]\n" + text[-target_chars:]

    def __repr__(self) -> str:
        return (
            f"ContextAssembler("
            f"max_tokens={self.max_total_tokens}, "
            f"slots={list(self._budgets.keys())})"
        )
