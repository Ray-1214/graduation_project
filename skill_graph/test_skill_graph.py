"""
Unit tests for SkillGraph.

Run with:
    python3 -m pytest skill_graph/test_skill_graph.py -v
    # or simply:
    python3 skill_graph/test_skill_graph.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skill_graph.skill_node import SkillNode
from skill_graph.skill_graph import SkillGraph

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []


def test(name: str):
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
print("  SkillGraph — Unit Tests")
print("=" * 60 + "\n")


# ── Helpers ──────────────────────────────────────────────────────────

def make_skill(name: str, utility: float = 0.0, **kw) -> SkillNode:
    sk = SkillNode(name=name, **kw)
    sk.utility = utility
    return sk


# ── 1. Node operations ──────────────────────────────────────────────
print("[Node Operations]")


@test("add_skill and get_skill")
def _():
    g = SkillGraph()
    sk = make_skill("alpha", utility=1.0)
    g.add_skill(sk)
    assert g.has_skill(sk.skill_id)
    assert g.get_skill(sk.skill_id) is sk
    assert len(g) == 1


@test("add_skill rejects duplicate")
def _():
    g = SkillGraph()
    sk = make_skill("alpha")
    g.add_skill(sk)
    try:
        g.add_skill(sk)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


@test("add_skill enforces capacity")
def _():
    g = SkillGraph(capacity=2)
    g.add_skill(make_skill("a"))
    g.add_skill(make_skill("b"))
    try:
        g.add_skill(make_skill("c"))
        assert False, "Should have raised OverflowError"
    except OverflowError:
        pass


@test("remove_skill removes node and edges")
def _():
    g = SkillGraph()
    a = make_skill("a")
    b = make_skill("b")
    g.add_skill(a)
    g.add_skill(b)
    g.add_edge(a.skill_id, b.skill_id, weight=1.0, edge_type="co_occurrence")
    g.remove_skill(a.skill_id)
    assert not g.has_skill(a.skill_id)
    assert len(g.get_edges()) == 0


# ── 2. Edge operations ──────────────────────────────────────────────
print("\n[Edge Operations]")


@test("add_edge with all three types")
def _():
    g = SkillGraph()
    a, b, c = make_skill("a"), make_skill("b"), make_skill("c")
    g.add_skill(a)
    g.add_skill(b)
    g.add_skill(c)
    g.add_edge(a.skill_id, b.skill_id, weight=0.8, edge_type="co_occurrence")
    g.add_edge(a.skill_id, c.skill_id, weight=1.0, edge_type="dependency")
    g.add_edge(b.skill_id, c.skill_id, weight=0.5, edge_type="abstraction")
    assert len(g.get_edges()) == 3
    assert len(g.get_edges(edge_type="abstraction")) == 1


@test("add_edge rejects non-positive weight")
def _():
    g = SkillGraph()
    a, b = make_skill("a"), make_skill("b")
    g.add_skill(a)
    g.add_skill(b)
    try:
        g.add_edge(a.skill_id, b.skill_id, weight=0.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


@test("add_edge rejects missing node")
def _():
    g = SkillGraph()
    a = make_skill("a")
    g.add_skill(a)
    try:
        g.add_edge(a.skill_id, "nonexistent", weight=1.0)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


@test("abstraction edges enforce DAG (reject cycle)")
def _():
    g = SkillGraph()
    a, b, c = make_skill("a"), make_skill("b"), make_skill("c")
    g.add_skill(a)
    g.add_skill(b)
    g.add_skill(c)
    g.add_edge(a.skill_id, b.skill_id, weight=1.0, edge_type="abstraction")
    g.add_edge(b.skill_id, c.skill_id, weight=1.0, edge_type="abstraction")
    # c → a would create a cycle in abstraction edges
    try:
        g.add_edge(c.skill_id, a.skill_id, weight=1.0, edge_type="abstraction")
        assert False, "Should have raised ValueError (DAG violation)"
    except ValueError as e:
        assert "cycle" in str(e).lower() or "DAG" in str(e)
    # The rejected edge should NOT remain in the graph
    assert len(g.get_edges(edge_type="abstraction")) == 2


@test("co_occurrence edges allow cycles")
def _():
    g = SkillGraph()
    a, b = make_skill("a"), make_skill("b")
    g.add_skill(a)
    g.add_skill(b)
    g.add_edge(a.skill_id, b.skill_id, weight=1.0, edge_type="co_occurrence")
    g.add_edge(b.skill_id, a.skill_id, weight=1.0, edge_type="co_occurrence")
    assert len(g.get_edges()) == 2  # no error


# ── 3. Active skills & matching ─────────────────────────────────────
print("\n[Queries]")


@test("get_active_skills filters by utility threshold")
def _():
    g = SkillGraph()
    a = make_skill("high", utility=0.8)
    b = make_skill("low", utility=0.2)
    c = make_skill("zero", utility=0.0)
    g.add_skill(a)
    g.add_skill(b)
    g.add_skill(c)
    active = g.get_active_skills(threshold=0.5)
    assert len(active) == 1
    assert active[0].name == "high"


@test("get_matching_skills uses initiation_set")
def _():
    g = SkillGraph()
    a = make_skill("calc", initiation_set=["math", "calculate"])
    b = make_skill("write", initiation_set=["essay", "writing"])
    g.add_skill(a)
    g.add_skill(b)
    matches = g.get_matching_skills("calculate 2 + 3")
    assert len(matches) == 1
    assert matches[0].name == "calc"


# ── 4. Structural entropy ───────────────────────────────────────────
print("\n[Structural Entropy]")


@test("entropy of empty graph is 0")
def _():
    g = SkillGraph()
    assert g.compute_entropy() == 0.0


@test("entropy of single skill is 0 (log₂(1) = 0)")
def _():
    g = SkillGraph()
    g.add_skill(make_skill("only", utility=1.0))
    assert abs(g.compute_entropy()) < 1e-9


@test("entropy of uniform distribution = log₂(n)")
def _():
    g = SkillGraph()
    n = 4
    for i in range(n):
        g.add_skill(make_skill(f"s{i}", utility=1.0))
    expected = math.log2(n)  # 2.0
    actual = g.compute_entropy()
    assert abs(actual - expected) < 1e-9, f"Expected {expected}, got {actual}"


@test("entropy of skewed distribution < log₂(n)")
def _():
    g = SkillGraph()
    g.add_skill(make_skill("dominant", utility=10.0))
    g.add_skill(make_skill("weak1", utility=0.1))
    g.add_skill(make_skill("weak2", utility=0.1))
    h = g.compute_entropy()
    max_h = math.log2(3)
    assert 0 < h < max_h, f"Expected 0 < {h} < {max_h}"


# ── 5. Capacity ─────────────────────────────────────────────────────
print("\n[Capacity]")


@test("compute_capacity counts skills above threshold")
def _():
    g = SkillGraph()
    g.add_skill(make_skill("a", utility=0.9))
    g.add_skill(make_skill("b", utility=0.6))
    g.add_skill(make_skill("c", utility=0.3))
    g.add_skill(make_skill("d", utility=0.0))
    assert g.compute_capacity(threshold=0.5) == 2
    assert g.compute_capacity(threshold=0.0) == 4


# ── 6. Decay ────────────────────────────────────────────────────────
print("\n[Decay]")


@test("decay_all reduces all utilities")
def _():
    g = SkillGraph()
    a = make_skill("a", utility=1.0)
    b = make_skill("b", utility=2.0)
    g.add_skill(a)
    g.add_skill(b)
    g.decay_all(gamma=0.1)
    assert abs(a.utility - 0.9) < 1e-9
    assert abs(b.utility - 1.8) < 1e-9


# ── 7. Subgraph & snapshot ──────────────────────────────────────────
print("\n[Subgraph & Snapshot]")


@test("get_subgraph returns induced subgraph")
def _():
    g = SkillGraph()
    a, b, c = make_skill("a"), make_skill("b"), make_skill("c")
    g.add_skill(a)
    g.add_skill(b)
    g.add_skill(c)
    g.add_edge(a.skill_id, b.skill_id, weight=1.0)
    g.add_edge(b.skill_id, c.skill_id, weight=1.0)
    sub = g.get_subgraph([a.skill_id, b.skill_id])
    assert len(sub) == 2
    assert sub.number_of_edges() == 1


@test("snapshot returns complete state dict")
def _():
    g = SkillGraph(capacity=50)
    a = make_skill("a", utility=1.0)
    b = make_skill("b", utility=2.0)
    g.add_skill(a)
    g.add_skill(b)
    g.add_edge(a.skill_id, b.skill_id, weight=0.5, edge_type="dependency")
    snap = g.snapshot()
    assert snap["num_skills"] == 2
    assert snap["num_edges"] == 1
    assert snap["capacity"] == 50
    assert "structural_entropy" in snap
    assert len(snap["nodes"]) == 2
    assert len(snap["edges"]) == 1
    assert snap["edges"][0]["edge_type"] == "dependency"


# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(results)
total = len(results)
if passed == total:
    print(f"  {PASS} All {total} tests passed!")
else:
    print(f"  {FAIL} {total - passed}/{total} tests failed.")
print("=" * 60 + "\n")

sys.exit(0 if passed == total else 1)
