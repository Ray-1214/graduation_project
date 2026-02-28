"""
Skill registry — discovers, registers, and dispatches tool calls.

Used by the ReAct reasoning loop to look up and execute tools
by name. All skills implement a common interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base skill interface
# ---------------------------------------------------------------------------

class BaseSkill(ABC):
    """Every skill must define a name, description, and execute method."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def execute(self, input_text: str) -> str:
        """Run the skill with the given input and return a string result."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Central registry of available skills."""

    def __init__(self) -> None:
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """Register a skill instance."""
        self._skills[skill.name] = skill
        logger.info("Registered skill: %s", skill.name)

    def get(self, name: str) -> Optional[BaseSkill]:
        """Look up a skill by name (case-insensitive)."""
        return self._skills.get(name) or self._skills.get(name.lower())

    def execute(self, name: str, input_text: str) -> str:
        """Look up and execute a skill. Returns error string if not found."""
        skill = self.get(name)
        if skill is None:
            msg = f"Unknown tool '{name}'. Available: {self.list_names()}"
            logger.warning(msg)
            return f"Error: {msg}"
        try:
            result = skill.execute(input_text)
            logger.info("Skill '%s' executed successfully.", name)
            return result
        except Exception as exc:
            logger.error("Skill '%s' failed: %s", name, exc)
            return f"Error executing {name}: {exc}"

    def list_names(self) -> list[str]:
        return list(self._skills.keys())

    def list_descriptions(self) -> str:
        """Format all skills as a string for prompt injection."""
        lines = []
        for skill in self._skills.values():
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={self.list_names()})"
