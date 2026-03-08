"""
Prompt templates for the CodingCoprocessor.

Each CodingTaskType maps to a structured English prompt optimised for
Qwen2.5-Coder-7B-Instruct.  Templates use ``str.format()`` placeholders:

    {description}   — natural-language task description
    {context_code}  — existing source code (may be empty)
    {constraints}   — formatted constraint list (may be empty)
    {language}      — target programming language
    {target_file}   — intended output file path (may be empty)
"""

from __future__ import annotations

from typing import Dict

from core.coding_coprocessor import CodingTaskType

# ═══════════════════════════════════════════════════════════════════
#  Prompt Templates
# ═══════════════════════════════════════════════════════════════════

CODING_PROMPTS: Dict[CodingTaskType, str] = {

    # ── WRITE_FUNCTION ────────────────────────────────────────────
    CodingTaskType.WRITE_FUNCTION: """\
You are an expert {language} programmer known for clean, production-ready code.

### Task
{description}

{context_code}
{constraints}

### Requirements
- Write the complete function inside a ```{language}``` code block.
- Include a clear docstring explaining purpose, arguments, and return value.
- Use full type hints on every parameter and the return type.
- Handle edge cases (empty input, None, type errors) with explicit checks.
- Do NOT include usage examples or a __main__ block.

### Output Format
```{language}```
<brief explanation of your design choices>
""",

    # ── EDIT_CODE ─────────────────────────────────────────────────
    CodingTaskType.EDIT_CODE: """\
You are an expert {language} programmer performing a targeted code edit.

### Requested Change
{description}

### Target File
{target_file}

### Original Code
```{language}
{context_code}
```

{constraints}

### Rules
- Output the COMPLETE modified file — never use "..." or "rest unchanged".
- Preserve existing imports, formatting conventions, and comments.
- Only change what is strictly required by the request.
- Place the full result inside a single ```{language}``` code block.

### Output Format
```{language}```
<summary of exactly what you changed and why>
""",

    # ── CODE_REVIEW ───────────────────────────────────────────────
    CodingTaskType.CODE_REVIEW: """\
You are a senior {language} code reviewer with deep expertise in software quality.

### Context
{description}

### Code to Review
```{language}
{context_code}
```

{constraints}

### Your review MUST include these sections (in order):

1. **BUGS** — List every correctness issue (logic errors, off-by-one, null \
dereferences, race conditions).  If none, write "None found."

2. **STYLE** — Naming conventions, formatting, docstring quality, type hint \
coverage.

3. **IMPROVEMENTS** — Performance optimisations, simplifications, better \
abstractions, or missing error handling.

4. **FIXED CODE** — If bugs were found, provide the corrected version in a \
```{language}``` code block.  If no bugs, omit this section.
""",

    # ── WRITE_TEST ────────────────────────────────────────────────
    CodingTaskType.WRITE_TEST: """\
You are an expert at writing exhaustive pytest test suites in {language}.

### Code Under Test
```{language}
{context_code}
```

### Testing Requirements
{description}

{constraints}

### Rules
- Use pytest (not unittest).
- Organise tests in a single test class or as top-level functions.
- Cover: happy-path, edge cases, error / exception paths, boundary values.
- Use ``@pytest.fixture`` where setup is reusable.
- Use ``pytest.raises`` for expected exceptions.
- Each test must have a single, clear assertion.
- Add a one-line docstring to every test function.

### Output Format
```{language}```
<brief explanation of test strategy>
""",

    # ── DEBUG ─────────────────────────────────────────────────────
    CodingTaskType.DEBUG: """\
You are an expert debugger.  Your job is to find and fix bugs precisely.

### Bug Report
{description}

### Buggy Code
```{language}
{context_code}
```

{constraints}

### Output (use EXACTLY this structure):

**ROOT CAUSE**
<one-paragraph explanation of why the bug occurs>

**FIX**
```{language}
<corrected code — full function / block, not just the changed line>
```

**EXPLANATION**
<step-by-step reasoning linking the root cause to your fix>
""",

    # ── EXPLAIN_CODE ──────────────────────────────────────────────
    CodingTaskType.EXPLAIN_CODE: """\
You are a patient and thorough programming teacher.

### Code
```{language}
{context_code}
```

### Focus
{description}

{constraints}

### Explain the code using these three sections:

1. **PURPOSE** — What the code does at a high level (1–2 sentences).

2. **STEP-BY-STEP** — Walk through the logic line by line or block by block.  \
Use numbered steps.  Mention variable state changes where helpful.

3. **DESIGN PATTERNS** — Identify any patterns used (factory, strategy, \
decorator, etc.) and explain WHY they are appropriate here.
""",

    # ── WRITE_SKILL (🔑 most important) ──────────────────────────
    CodingTaskType.WRITE_SKILL: """\
You are writing a new skill module for a self-evolving AI agent.

### BaseSkill Interface
Every skill MUST inherit from ``BaseSkill`` and implement these members:

```python
from abc import ABC, abstractmethod

class BaseSkill(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        \"\"\"Unique identifier for this skill.\"\"\"
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        \"\"\"One-line description shown to the agent planner.\"\"\"
        ...

    @abstractmethod
    def execute(self, input_text: str) -> str:
        \"\"\"Run the skill.

        Args:
            input_text: Natural-language instruction from the agent.

        Returns:
            The skill's output as a string.
        \"\"\"
        ...
```

### Skill Description
{description}

### Reference Skills (for style)
{context_code}

{constraints}

### Requirements
- Inherit from ``BaseSkill`` (import from ``skills.base_skill``).
- Implement ``name``, ``description``, and ``execute`` exactly as above.
- The skill must work fully offline — no network calls.
- Handle all errors gracefully: catch exceptions inside ``execute`` and \
return a human-readable error string instead of raising.
- Include a module-level docstring, class docstring, and inline comments.
- Use logging (``import logging; logger = logging.getLogger(__name__)``).
- Place the complete, ready-to-save file inside a ```{language}``` block.

### Output Format
```{language}```
<brief explanation of the skill's design>
""",
}


# ═══════════════════════════════════════════════════════════════════
#  Public accessor
# ═══════════════════════════════════════════════════════════════════

def get_coding_prompt(task_type: CodingTaskType) -> str:
    """Return the prompt template for *task_type*.

    Args:
        task_type: A :class:`CodingTaskType` enum member.

    Returns:
        The format-string template.

    Raises:
        KeyError: If no template is registered for *task_type*.
    """
    try:
        return CODING_PROMPTS[task_type]
    except KeyError:
        raise KeyError(
            f"No prompt template registered for {task_type!r}. "
            f"Available types: {list(CODING_PROMPTS.keys())}"
        ) from None
