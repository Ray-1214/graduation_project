"""
Unit tests for MemoryPartition.

Run with:
    python3 skill_graph/test_memory_partition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skill_graph.skill_node import SkillNode
from skill_graph.skill_graph import SkillGraph
from skill_graph.memory_partition import MemoryPartition

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


def make_skill(name: str, utility: float = 0.0) -> SkillNode:
    sk = SkillNode(name=name)
    sk.utility = utility
    return sk


# Thresholds: θ_high=0.7, θ_low=0.3, ε_h=0.1, ε_l=0.1
# Promotion to active:   U ≥ 0.8  (0.7 + 0.1)
# Demotion from active:  U < 0.6  (0.7 - 0.1)
# Promotion from archive: U > 0.4  (0.3 + 0.1)
# Demotion to archive:    U ≤ 0.2  (0.3 - 0.1)


print("\n" + "=" * 60)
print("  MemoryPartition — Unit Tests")
print("=" * 60 + "\n")


# ── 1. Basic tier assignment ─────────────────────────────────────────
print("[Basic Tier Assignment]")


@test("new cold skill with high utility → active")
def _():
    mp = MemoryPartition()
    sk = make_skill("high", utility=0.9)
    tier = mp.assign_tier(sk, "cold")
    assert tier == "active", f"Expected active, got {tier}"


@test("new cold skill with medium utility → stays cold")
def _():
    mp = MemoryPartition()
    sk = make_skill("mid", utility=0.5)
    tier = mp.assign_tier(sk, "cold")
    assert tier == "cold", f"Expected cold, got {tier}"


@test("new cold skill with low utility → archive")
def _():
    mp = MemoryPartition()
    sk = make_skill("low", utility=0.1)
    tier = mp.assign_tier(sk, "cold")
    assert tier == "archive", f"Expected archive, got {tier}"


@test("archive skill with high utility → direct jump to active")
def _():
    mp = MemoryPartition()
    sk = make_skill("revived", utility=0.85)
    tier = mp.assign_tier(sk, "archive")
    assert tier == "active", f"Expected active, got {tier}"


@test("active skill with very low utility → skip cold, go to archive")
def _():
    mp = MemoryPartition()
    sk = make_skill("crashed", utility=0.1)
    tier = mp.assign_tier(sk, "active")
    assert tier == "archive", f"Expected archive, got {tier}"


# ── 2. Hysteresis: active boundary ──────────────────────────────────
print("\n[Hysteresis: Active ↔ Cold Boundary]")


@test("active skill at U=0.65 stays active (hysteresis protects)")
def _():
    # Promotion threshold is 0.8, but demotion is 0.6
    # At 0.65, skill is ABOVE demotion threshold → stays active
    mp = MemoryPartition()
    sk = make_skill("borderline", utility=0.65)
    tier = mp.assign_tier(sk, "active")
    assert tier == "active", f"Expected active, got {tier}"


@test("cold skill at U=0.65 stays cold (below promotion threshold)")
def _():
    # Same utility, but from cold side: needs ≥ 0.8 to promote
    mp = MemoryPartition()
    sk = make_skill("borderline", utility=0.65)
    tier = mp.assign_tier(sk, "cold")
    assert tier == "cold", f"Expected cold, got {tier}"


@test("active skill at U=0.55 demotes to cold")
def _():
    # Below demotion threshold (0.6) → drops to cold
    mp = MemoryPartition()
    sk = make_skill("declining", utility=0.55)
    tier = mp.assign_tier(sk, "active")
    assert tier == "cold", f"Expected cold, got {tier}"


@test("hysteresis dead band: U=0.75 — cold can't promote, active won't demote")
def _():
    # U=0.75 is in the dead band [0.6, 0.8)
    mp = MemoryPartition()
    sk = make_skill("deadband", utility=0.75)
    assert mp.assign_tier(sk, "active") == "active"   # stays
    assert mp.assign_tier(sk, "cold") == "cold"        # stays


# ── 3. Hysteresis: archive boundary ──────────────────────────────────
print("\n[Hysteresis: Cold ↔ Archive Boundary]")


@test("archive skill at U=0.35 stays archive (below promotion threshold)")
def _():
    # Promotion from archive needs U > 0.4 (θ_low + ε_l)
    mp = MemoryPartition()
    sk = make_skill("stale", utility=0.35)
    tier = mp.assign_tier(sk, "archive")
    assert tier == "archive", f"Expected archive, got {tier}"


@test("archive skill at U=0.45 promotes to cold")
def _():
    # U=0.45 > 0.4 → promote to cold
    mp = MemoryPartition()
    sk = make_skill("recovering", utility=0.45)
    tier = mp.assign_tier(sk, "archive")
    assert tier == "cold", f"Expected cold, got {tier}"


@test("cold skill at U=0.25 stays cold (above archive threshold)")
def _():
    # Demotion to archive needs U ≤ 0.2 (θ_low - ε_l)
    mp = MemoryPartition()
    sk = make_skill("cold_ok", utility=0.25)
    tier = mp.assign_tier(sk, "cold")
    assert tier == "cold", f"Expected cold, got {tier}"


@test("cold skill at U=0.15 demotes to archive")
def _():
    mp = MemoryPartition()
    sk = make_skill("dying", utility=0.15)
    tier = mp.assign_tier(sk, "cold")
    assert tier == "archive", f"Expected archive, got {tier}"


# ── 4. Anti-oscillation scenario ────────────────────────────────────
print("\n[Anti-Oscillation Scenario]")


@test("skill cannot oscillate at active boundary (full cycle)")
def _():
    """
    Simulate a skill whose utility fluctuates around θ_high=0.7:
      Episode 1: U = 0.85 → enters active (≥ 0.8)
      Episode 2: U = 0.72 → stays active (hysteresis, ≥ 0.6)
      Episode 3: U = 0.65 → STILL active (≥ 0.6)
      Episode 4: U = 0.58 → drops to cold (< 0.6)
      Episode 5: U = 0.72 → stays cold (< 0.8, can't re-promote)
      Episode 6: U = 0.82 → re-enters active (≥ 0.8)

    Without hysteresis, episodes 2-3 and 5 would cause oscillation.
    """
    mp = MemoryPartition()
    sk = make_skill("oscillator")

    trajectory = [
        (0.85, "active"),
        (0.72, "active"),   # hysteresis keeps it
        (0.65, "active"),   # still protected
        (0.58, "cold"),     # finally drops
        (0.72, "cold"),     # can't re-promote yet
        (0.82, "active"),   # crosses promotion threshold
    ]

    current_tier = "cold"
    for i, (utility, expected_tier) in enumerate(trajectory):
        sk.utility = utility
        new_tier = mp.assign_tier(sk, current_tier)
        assert new_tier == expected_tier, (
            f"Episode {i+1}: U={utility}, "
            f"expected {expected_tier}, got {new_tier} "
            f"(from {current_tier})"
        )
        current_tier = new_tier


@test("skill cannot oscillate at archive boundary (full cycle)")
def _():
    """
    Simulate a skill whose utility fluctuates around θ_low=0.3:
      Episode 1: U = 0.10 → enters archive (≤ 0.2)
      Episode 2: U = 0.25 → stays archive (hysteresis, ≤ 0.4)
      Episode 3: U = 0.38 → stays archive (≤ 0.4)
      Episode 4: U = 0.45 → promotes to cold (> 0.4)
      Episode 5: U = 0.35 → stays cold (> 0.2)
      Episode 6: U = 0.15 → drops to archive (≤ 0.2)
    """
    mp = MemoryPartition()
    sk = make_skill("archive_oscillator")

    trajectory = [
        (0.10, "archive"),
        (0.25, "archive"),  # hysteresis keeps it
        (0.38, "archive"),  # still under promotion threshold
        (0.45, "cold"),     # crosses promotion threshold
        (0.35, "cold"),     # can't re-demote yet
        (0.15, "archive"),  # crosses demotion threshold
    ]

    current_tier = "cold"
    for i, (utility, expected_tier) in enumerate(trajectory):
        sk.utility = utility
        new_tier = mp.assign_tier(sk, current_tier)
        assert new_tier == expected_tier, (
            f"Episode {i+1}: U={utility}, "
            f"expected {expected_tier}, got {new_tier} "
            f"(from {current_tier})"
        )
        current_tier = new_tier


# ── 5. Bulk update with SkillGraph ───────────────────────────────────
print("\n[Bulk Update with SkillGraph]")


@test("update_all partitions graph correctly")
def _():
    g = SkillGraph()
    a = make_skill("active_skill", utility=0.9)
    b = make_skill("cold_skill", utility=0.5)
    c = make_skill("archive_skill", utility=0.1)
    g.add_skill(a)
    g.add_skill(b)
    g.add_skill(c)

    mp = MemoryPartition()
    result = mp.update_all(g)

    assert result[a.skill_id] == "active"
    assert result[b.skill_id] == "cold"
    assert result[c.skill_id] == "archive"

    assert mp.get_tier(a.skill_id) == "active"
    assert set(mp.get_skills_by_tier("active")) == {a.skill_id}
    assert set(mp.get_skills_by_tier("cold")) == {b.skill_id}
    assert set(mp.get_skills_by_tier("archive")) == {c.skill_id}


@test("update_all preserves hysteresis across calls")
def _():
    g = SkillGraph()
    sk = make_skill("tracked", utility=0.85)
    g.add_skill(sk)

    mp = MemoryPartition()
    mp.update_all(g)
    assert mp.get_tier(sk.skill_id) == "active"

    # Utility drops but stays in hysteresis band
    sk.utility = 0.65
    mp.update_all(g)
    assert mp.get_tier(sk.skill_id) == "active"  # protected

    # Utility drops below demotion threshold
    sk.utility = 0.55
    mp.update_all(g)
    assert mp.get_tier(sk.skill_id) == "cold"


@test("summary counts are correct")
def _():
    g = SkillGraph()
    g.add_skill(make_skill("a1", utility=0.9))
    g.add_skill(make_skill("a2", utility=0.85))
    g.add_skill(make_skill("c1", utility=0.5))
    g.add_skill(make_skill("ar1", utility=0.1))

    mp = MemoryPartition()
    mp.update_all(g)
    s = mp.summary()
    assert s["active"] == 2
    assert s["cold"] == 1
    assert s["archive"] == 1


# ── 6. Edge cases ───────────────────────────────────────────────────
print("\n[Edge Cases]")


@test("theta_high <= theta_low raises ValueError")
def _():
    try:
        MemoryPartition(theta_high=0.3, theta_low=0.7)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


@test("skill at exact boundary: U = θ_high + ε_h (=0.8) → active")
def _():
    mp = MemoryPartition()
    sk = make_skill("exact", utility=0.8)
    assert mp.assign_tier(sk, "cold") == "active"


@test("skill at exact boundary: U = 0.19 (< θ_low − ε_l) → archive")
def _():
    mp = MemoryPartition()
    sk = make_skill("exact_low", utility=0.19)
    assert mp.assign_tier(sk, "cold") == "archive"


@test("untracked skill defaults to cold")
def _():
    mp = MemoryPartition()
    assert mp.get_tier("nonexistent") == "cold"


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
