"""
File operations skill — sandboxed file read/write.
"""

from __future__ import annotations

from pathlib import Path
from skills.registry import BaseSkill


class FileOps(BaseSkill):
    """Read and write files within a sandboxed workspace directory."""

    def __init__(self, workspace_dir: str = ".") -> None:
        self._workspace = Path(workspace_dir).resolve()

    @property
    def name(self) -> str:
        return "file_ops"

    @property
    def description(self) -> str:
        return (
            "Read or write files. "
            "Format: 'read <path>' or 'write <path> <content>'"
        )

    def _resolve_safe(self, path_str: str) -> Path:
        """Resolve a path and ensure it stays within the workspace."""
        target = (self._workspace / path_str).resolve()
        if not str(target).startswith(str(self._workspace)):
            raise PermissionError(
                f"Path '{path_str}' escapes the workspace directory."
            )
        return target

    def execute(self, input_text: str) -> str:
        parts = input_text.strip().split(maxsplit=2)
        if len(parts) < 2:
            return "Error: expected 'read <path>' or 'write <path> <content>'"

        action = parts[0].lower()

        if action == "read":
            return self._read(parts[1])
        elif action == "write" and len(parts) == 3:
            return self._write(parts[1], parts[2])
        else:
            return "Error: expected 'read <path>' or 'write <path> <content>'"

    def _read(self, path_str: str) -> str:
        try:
            target = self._resolve_safe(path_str)
            if not target.exists():
                return f"Error: file not found: {path_str}"
            content = target.read_text(encoding="utf-8", errors="replace")
            # Truncate very large files
            if len(content) > 4000:
                return content[:4000] + "\n... (truncated)"
            return content
        except Exception as exc:
            return f"Error: {exc}"

    def _write(self, path_str: str, content: str) -> str:
        try:
            target = self._resolve_safe(path_str)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} chars to {path_str}"
        except Exception as exc:
            return f"Error: {exc}"
