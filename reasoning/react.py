"""
ReAct — Reasoning + Acting loop.

Implements the Thought → Action → Observation loop from the ReAct
paper (Yao et al., 2023). The LLM reasons about what tool to use,
the framework dispatches the tool call, and the observation is fed
back for the next iteration.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from core.llm_interface import BaseLLM
from core.prompt_builder import PromptBuilder
from memory.episodic_log import EpisodicLog
from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class ReActLoop:
    """Thought → Action → Observation reasoning loop with tool use."""

    def __init__(
        self,
        llm: BaseLLM,
        prompt_builder: PromptBuilder,
        skill_registry: SkillRegistry,
        max_steps: int = 8,
    ) -> None:
        self.llm = llm
        self.pb = prompt_builder
        self.skills = skill_registry
        self.max_steps = max_steps

    def run(self, task: str, episode: Optional[EpisodicLog] = None) -> str:
        """Execute the ReAct loop until Finish or max_steps."""
        previous_steps = ""
        tool_descriptions = self.skills.list_descriptions()

        for step_num in range(1, self.max_steps + 1):
            logger.info("ReAct step %d/%d", step_num, self.max_steps)

            # Build prompt with accumulated history
            prompt = self.pb.build(
                "react",
                task=task,
                tool_descriptions=tool_descriptions,
                previous_steps=previous_steps,
            )

            # Get LLM response
            response = self.llm.generate(prompt, stop=["\nObservation:"])
            response = response.strip()

            if episode:
                episode.log_step("thought", response)

            # Check for Finish
            finish_match = re.search(r"Finish\[(.+?)\]", response, re.DOTALL)
            if finish_match:
                answer = finish_match.group(1).strip()
                logger.info("ReAct finished at step %d: %s", step_num, answer[:80])
                if episode:
                    episode.log_step("finish", answer)
                return answer

            # Parse Action
            action_match = re.search(
                r"Action\[(\w+)\]:\s*(.+?)(?:\n|$)", response, re.DOTALL
            )
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()

                if episode:
                    episode.log_step("action", f"{tool_name}: {tool_input}")

                # Execute tool
                observation = self.skills.execute(tool_name, tool_input)

                if episode:
                    episode.log_step("observation", observation)

                # Append to history
                previous_steps += (
                    f"{response}\n"
                    f"Observation: {observation}\n\n"
                )
            else:
                # No action parsed — append response as is and continue
                logger.warning(
                    "ReAct step %d: no Action or Finish found in response.",
                    step_num,
                )
                previous_steps += f"{response}\n\n"

        # Max steps reached
        fallback = (
            f"(ReAct loop reached max steps ({self.max_steps}). "
            f"Last response: {response[:200]})"
        )
        if episode:
            episode.log_step("error", fallback)
        return fallback
