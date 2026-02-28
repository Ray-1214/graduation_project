"""
Web search skill — stub implementation (no external APIs).

Returns a placeholder response. Designed to be swapped for a
real search backend (e.g. SearXNG, local index) when available.
"""

from __future__ import annotations

from skills.registry import BaseSkill


class WebSearch(BaseSkill):
    """Stub web search — returns placeholder results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information (currently returns stub results)"

    def execute(self, input_text: str) -> str:
        query = input_text.strip()
        return (
            f"[Web Search Stub] No external API configured.\n"
            f"Query: '{query}'\n"
            f"To enable real search, implement a backend (e.g. SearXNG, "
            f"local Elasticsearch) and replace this stub."
        )
