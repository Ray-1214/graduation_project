"""Pytest configuration — ensures project root is importable."""

import sys
from pathlib import Path

# Add project root to sys.path so `from skill_graph.X import Y` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
