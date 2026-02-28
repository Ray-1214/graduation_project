"""
Smoke test — verifies that all modules import correctly and
basic non-LLM functionality works.

Run with:
    python experiments/smoke_test.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []


def test(name: str):
    """Decorator to register and run a test."""
    def decorator(fn):
        try:
            fn()
            print(f"  {PASS} {name}")
            results.append(True)
        except Exception as exc:
            print(f"  {FAIL} {name}: {exc}")
            results.append(False)
        return fn
    return decorator


print("\n" + "=" * 60)
print("  Cognitive Architecture — Smoke Tests")
print("=" * 60 + "\n")


# ── 1. Core imports ──────────────────────────────────────────────────
print("[Core]")

@test("Config loads with defaults")
def _():
    from core.config import Config
    c = Config.default()
    assert c.n_gpu_layers == 99
    assert c.n_ctx == 4096
    assert "mistral" in c.model_path.lower()

@test("PromptBuilder renders templates")
def _():
    from core.prompt_builder import PromptBuilder
    pb = PromptBuilder()
    prompt = pb.build("cot", task="test task")
    assert "test task" in prompt
    assert "step by step" in prompt

@test("PromptBuilder renders all templates without error")
def _():
    from core.prompt_builder import PromptBuilder
    pb = PromptBuilder()
    # Check each template has at least the right variables
    pb.build("cot", task="t")
    pb.build("tot_expand", task="t", current_path="p", n_branches=3)
    pb.build("tot_evaluate", task="t", reasoning_path="p")
    pb.build("react", task="t", tool_descriptions="d", previous_steps="")
    pb.build("reflexion", task="t", trajectory="tr", outcome="o", score="0.5")
    pb.build("rag_context", task="t", retrieved_context="c")
    pb.build("evaluate", task="t", answer="a")

@test("LlamaCppLLM class exists (lazy, no model load)")
def _():
    from core.llm_interface import LlamaCppLLM, BaseLLM
    assert issubclass(LlamaCppLLM, BaseLLM)


# ── 2. Memory ────────────────────────────────────────────────────────
print("\n[Memory]")

@test("ShortTermMemory ring buffer works")
def _():
    from memory.short_term import ShortTermMemory
    stm = ShortTermMemory(capacity=3)
    stm.add("user", "hello")
    stm.add("assistant", "hi")
    stm.add("user", "how?")
    stm.add("assistant", "fine")  # should evict "hello"
    assert len(stm) == 3
    ctx = stm.get_context()
    assert "hello" not in ctx
    assert "fine" in ctx

@test("LongTermMemory stores and retrieves")
def _():
    import tempfile, os
    from memory.long_term import LongTermMemory, ReflectionEntry
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        ltm = LongTermMemory(path)
        entry = ReflectionEntry(
            task="math problem",
            reflection="Should check units",
            lessons=["always verify units", "double-check arithmetic"],
            score=0.7,
        )
        ltm.store(entry)
        assert len(ltm) == 1
        results_ = ltm.retrieve(["math", "units"], top_k=1)
        assert len(results_) == 1
        assert results_[0].task == "math problem"
    finally:
        os.unlink(path)

@test("EpisodicLog records and serializes")
def _():
    from memory.episodic_log import EpisodicLog
    ep = EpisodicLog(task="test", strategy="cot")
    ep.log_step("thought", "thinking …")
    ep.log_step("finish", "answer")
    ep.finish(result="answer", score=0.9)
    d = ep.to_dict()
    assert d["task"] == "test"
    assert len(d["steps"]) == 2
    assert d["score"] == 0.9

@test("convert_log_to_trace pairs action→outcome correctly (ReAct)")
def _():
    from memory.episodic_log import EpisodicLog, convert_log_to_trace
    ep = EpisodicLog(task="What is 2+3?", strategy="react")
    ep.log_step("thought", "I should use the calculator")
    ep.log_step("action", "calculator: 2+3")
    ep.log_step("observation", "5")
    ep.log_step("thought", "I have the answer")
    ep.log_step("finish", "The answer is 5")
    ep.finish(result="The answer is 5", score=0.8)
    trace = convert_log_to_trace(ep, task_id="test-react")
    assert trace.task_id == "test-react"
    assert trace.strategy == "react"
    assert trace.success is True
    assert trace.score == 0.8
    # thought→(no outcome), action→observation, thought→finish = 3 trace steps
    assert len(trace.steps) == 3
    # Second step should pair action→observation
    assert "calculator" in trace.steps[1].action
    assert "5" in trace.steps[1].outcome

@test("convert_log_to_trace handles single-thought CoT")
def _():
    from memory.episodic_log import EpisodicLog, convert_log_to_trace
    ep = EpisodicLog(task="Explain gravity", strategy="cot")
    ep.log_step("thought", "Using CoT")
    ep.log_step("finish", "Gravity is a force")
    ep.finish(result="Gravity is a force", score=0.6)
    trace = convert_log_to_trace(ep)
    assert len(trace.steps) == 1
    assert trace.success is True
    assert trace.task_id  # auto-generated

@test("EpisodicTrace serializes to dict")
def _():
    from memory.episodic_log import EpisodicLog, convert_log_to_trace
    ep = EpisodicLog(task="test", strategy="cot")
    ep.log_step("thought", "think")
    ep.log_step("finish", "done")
    ep.finish(result="done", score=0.9)
    trace = convert_log_to_trace(ep, task_id="ser-test")
    d = trace.to_dict()
    assert d["task_id"] == "ser-test"
    assert d["success"] is True
    assert len(d["steps"]) == 1
    assert "state" in d["steps"][0]
    assert "action" in d["steps"][0]
    assert "outcome" in d["steps"][0]


# ── 3. Skills ─────────────────────────────────────────────────────────
print("\n[Skills]")

@test("Calculator evaluates arithmetic safely")
def _():
    from skills.calculator import Calculator
    calc = Calculator()
    assert calc.execute("25 * 17") == "425"
    assert calc.execute("2 + 3 * 4") == "14"
    assert calc.execute("100 / 4") == "25"
    # Dangerous input should fail safely
    result = calc.execute("__import__('os').system('ls')")
    assert "Error" in result

@test("SkillRegistry dispatches correctly")
def _():
    from skills.registry import SkillRegistry
    from skills.calculator import Calculator
    reg = SkillRegistry()
    reg.register(Calculator())
    assert reg.execute("calculator", "2 + 3") == "5"
    assert "Error" in reg.execute("nonexistent", "foo")

@test("FileOps blocks path traversal")
def _():
    from skills.file_ops import FileOps
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fo = FileOps(workspace_dir=td)
        result = fo.execute("read ../../../etc/passwd")
        assert "Error" in result or "escapes" in result.lower() or "not found" in result.lower()


# ── 4. RAG ────────────────────────────────────────────────────────────
print("\n[RAG]")

@test("Indexer chunks text correctly")
def _():
    from rag.indexer import Indexer
    from memory.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)
    vs._texts = []
    vs._metadatas = []
    idx = Indexer(vs, chunk_size=100, chunk_overlap=20)
    chunks = idx._chunk_text("A" * 250, source="test.txt")
    assert len(chunks) >= 2
    assert all(len(c["text"]) <= 100 for c in chunks)


# ── 5. Reasoning ──────────────────────────────────────────────────────
print("\n[Reasoning]")

@test("StrategyPlanner selects strategies by keyword")
def _():
    from reasoning.planner import StrategyPlanner
    sp = StrategyPlanner()
    assert sp.select_strategy("calculate 2+3") == "react"
    assert sp.select_strategy("brainstorm creative ideas") == "tot"
    assert sp.select_strategy("explain gravity") == "cot"

@test("ThoughtNode tracks path correctly")
def _():
    from reasoning.tot import ThoughtNode
    root = ThoughtNode(thought="Root")
    child = ThoughtNode(thought="Step 1", parent=root)
    grandchild = ThoughtNode(thought="Step 2", parent=child)
    path = grandchild.path_str()
    assert "Root" in path
    assert "Step 1" in path
    assert "Step 2" in path

@test("Reflexion extracts lessons from text")
def _():
    from reasoning.reflexion import Reflexion
    text = "1. Check units\n2. Verify bounds\n3. Re-read the question"
    lessons = Reflexion._extract_lessons(text)
    assert len(lessons) == 3
    assert "Check units" in lessons[0]


# ── 6. Agents ─────────────────────────────────────────────────────────
print("\n[Agents]")

@test("MainAgent can be imported")
def _():
    from agents.main_agent import MainAgent, AgentResult

@test("EvaluatorAgent can be imported")
def _():
    from agents.evaluator_agent import EvaluatorAgent


# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(results)
total = len(results)
if passed == total:
    print(f"  {PASS} All {total} tests passed!")
else:
    failed = total - passed
    print(f"  {FAIL} {failed}/{total} tests failed.")
print("=" * 60 + "\n")

sys.exit(0 if passed == total else 1)
