"""
Calculator skill — safe arithmetic evaluation.
"""

from __future__ import annotations

import ast
import operator
from skills.registry import BaseSkill


# Safe operators for arithmetic evaluation
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only safe arithmetic ops."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    else:
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class Calculator(BaseSkill):
    """Evaluates arithmetic expressions safely (no exec/eval)."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a mathematical expression (e.g. '25 * 17 + 3')"

    def execute(self, input_text: str) -> str:
        try:
            tree = ast.parse(input_text.strip(), mode="eval")
            result = _safe_eval(tree)
            # Format nicely: drop .0 for integers
            if isinstance(result, float) and result == int(result):
                return str(int(result))
            return str(result)
        except Exception as exc:
            return f"Error: {exc}"
