"""
Prompt template engine for all reasoning strategies.

Uses Python f-string style templates (no Jinja2 dependency).
Each template is keyed by strategy name and rendered via
PromptBuilder.build(name, **variables).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, str] = {

    # ── Chain of Thought (single-pass baseline) ──────────────────────────
    "cot": (
        "Solve the following problem step by step.\n\n"
        "Problem: {task}\n\n"
        "Let's think step by step:\n"
    ),

    # ── Tree of Thoughts: branch expansion ───────────────────────────────
    "tot_expand": (
        "Given the problem: {task}\n\n"
        "Current reasoning path:\n{current_path}\n\n"
        "Generate {n_branches} distinct next reasoning steps. "
        "Each step should explore a DIFFERENT approach or hypothesis.\n\n"
        "Format your response as:\n"
        "Step 1: [reasoning step]\n"
        "Step 2: [reasoning step]\n"
        "Step 3: [reasoning step]\n"
    ),

    # ── Tree of Thoughts: node evaluation ────────────────────────────────
    "tot_evaluate": (
        "Given the problem: {task}\n\n"
        "Evaluate the following reasoning path on a scale of 1-10.\n"
        "Consider correctness, completeness, and logical coherence.\n\n"
        "Reasoning path:\n{reasoning_path}\n\n"
        "Score (just the number): "
    ),

    # ── ReAct loop ───────────────────────────────────────────────────────
    "react": (
        "Answer the following question using the available tools.\n\n"
        "Question: {task}\n\n"
        "Available tools:\n{tool_descriptions}\n\n"
        "Strategy:\n"
        "- If you are unsure about the answer, first use web_search to look it up.\n"
        "- If web_search results are insufficient or unclear, use admin_query to ask the administrator.\n"
        "- Prefer using existing knowledge before making external calls.\n\n"
        "Use this EXACT format for each step:\n"
        "Thought: [your reasoning about what to do next]\n"
        "Action[tool_name]: [tool input]\n\n"
        "After a tool is used you will see:\n"
        "Observation: [tool result]\n\n"
        "When you have the final answer:\n"
        "Thought: I now have enough information.\n"
        "Finish[your final answer]\n\n"
        "{previous_steps}"
    ),

    # ── Reflexion: self-reflection writing ───────────────────────────────
    "reflexion": (
        "You completed a task. Reflect on your performance.\n\n"
        "Task: {task}\n\n"
        "Your trajectory:\n{trajectory}\n\n"
        "Outcome: {outcome}\n"
        "Score: {score}\n\n"
        "Write a concise self-reflection:\n"
        "1. What went well?\n"
        "2. What went wrong?\n"
        "3. What specific lessons should you remember for similar future tasks?\n\n"
        "Reflection:\n"
    ),

    # ── RAG: context injection ───────────────────────────────────────────
    "rag_context": (
        "Use the following retrieved context to help answer the question.\n"
        "If the context is not relevant, rely on your own knowledge.\n\n"
        "--- Retrieved Context ---\n"
        "{retrieved_context}\n"
        "--- End Context ---\n\n"
        "Question: {task}\n\n"
        "Answer:\n"
    ),

    # ── Evaluator: LLM-as-judge ──────────────────────────────────────────
    "evaluate": (
        "You are an impartial evaluator. Score the following answer "
        "to the given task on a scale from 0.0 to 1.0.\n\n"
        "Task: {task}\n\n"
        "Answer: {answer}\n\n"
        "Consider: correctness, completeness, clarity, and relevance.\n\n"
        "Score (a single decimal number between 0.0 and 1.0): "
    ),
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Renders prompt templates with variable substitution."""

    def __init__(self, extra_templates: Optional[Dict[str, str]] = None) -> None:
        self.templates: Dict[str, str] = {**TEMPLATES}
        if extra_templates:
            self.templates.update(extra_templates)

    def build(self, template_name: str, **variables: Any) -> str:
        """Render a named template with the given variables.

        Raises KeyError if the template name is unknown.
        Raises KeyError if a required variable is missing.
        """
        if template_name not in self.templates:
            raise KeyError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(self.templates.keys())}"
            )
        template = self.templates[template_name]
        try:
            return template.format(**variables)
        except KeyError as exc:
            raise KeyError(
                f"Missing variable {exc} for template '{template_name}'"
            ) from exc

    def register(self, name: str, template: str) -> None:
        """Register a new template (or override an existing one)."""
        self.templates[name] = template

    def list_templates(self) -> list[str]:
        """Return all registered template names."""
        return list(self.templates.keys())
