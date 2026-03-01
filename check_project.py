#!/usr/bin/env python3
"""
🔍 Self-Evolving Skill Graph — 專案檢查報告

用法: python inspect.py
一鍵印出整個專案狀態摘要（Phase 進度、測試結果、知識庫、技能圖、程式碼統計）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── 常數 ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent

PHASES = [
    ("Phase 0",   "Trace 格式確認",           None),
    ("Phase 1",   "Skill Graph 資料結構",     "tests/test_skill_graph.py"),
    ("Phase 1.5", "主動知識獲取",              "tests/test_knowledge.py"),
    ("Phase 2",   "Evolution Operator Φ",     "tests/test_evolution.py"),
    ("Phase 3",   "接進主代理循環",            "tests/test_integration.py"),
    ("Phase 4",   "指標追蹤與實驗",            "tests/test_metrics.py"),
    ("Phase 5",   "前端視覺化",               None),
]

CODE_CATEGORIES = [
    ("Phase 1 (Skill Graph)",  ["skill_graph/*.py"]),
    ("Phase 1.5 (Knowledge)",  ["rag/knowledge_store.py",
                                "skills/web_search.py",
                                "skills/admin_query.py"]),
    ("推理引擎",                ["reasoning/*.py"]),
    ("Agent 核心",              ["agents/*.py", "core/*.py"]),
    ("Skills (工具層)",         ["skills/*.py"]),
    ("測試",                    ["tests/*.py"]),
]

KNOWLEDGE_STORE_PATH = ROOT / "rag" / "knowledge_base" / "knowledge_store.json"
SKILL_GRAPH_PATH = ROOT / "data" / "skill_graph.json"

W = 60  # column width for alignment


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def banner(icon: str, title: str) -> None:
    print()
    print(f"{'═' * W}")
    print(f"  {icon} {title}")
    print(f"{'═' * W}")


def run_pytest(test_file: str) -> dict:
    """Run pytest on a single file and return parsed results.

    Returns dict with keys: passed, failed, errors, output, failures.
    """
    filepath = ROOT / test_file
    if not filepath.exists():
        return {"exists": False}

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(filepath), "-q", "--tb=line"],
        capture_output=True, text=True, cwd=str(ROOT),
        timeout=120,
    )

    output = result.stdout + result.stderr
    passed = failed = errors = 0

    # Parse summary line like "15 passed" or "3 failed, 12 passed"
    m = re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", output)
    if m:
        errors = int(m.group(1))

    # Extract FAILED test names
    failures = re.findall(r"FAILED (.+?)(?:\s|$)", output)

    return {
        "exists": True,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "failures": failures,
        "output": output,
    }


def count_lines(filepath: Path) -> int:
    """Count non-empty lines in a Python file."""
    try:
        return sum(1 for line in filepath.read_text(encoding="utf-8").splitlines()
                   if line.strip())
    except Exception:
        return 0


def resolve_globs(patterns: list[str]) -> list[Path]:
    """Resolve glob patterns relative to ROOT, excluding __pycache__ and tiny __init__.py."""
    files: list[Path] = []
    for pat in patterns:
        for p in ROOT.glob(pat):
            if "__pycache__" in str(p):
                continue
            if p.name == "__init__.py" and count_lines(p) <= 2:
                continue
            if p.is_file():
                files.append(p)
    return sorted(set(files))


# ═══════════════════════════════════════════════════════════════════
#  區塊 1: Phase 進度總覽
# ═══════════════════════════════════════════════════════════════════

def section_phases(test_results: dict[str, dict]) -> None:
    banner("📊", "Phase 進度總覽")
    print()
    for phase_name, desc, test_file in PHASES:
        # Phase 0: always done
        if phase_name == "Phase 0":
            status = "✅ 已完成"
        # Phase 5: check for frontend files
        elif phase_name == "Phase 5":
            frontend = list(ROOT.glob("frontend/*")) + list(ROOT.glob("static/*"))
            status = "✅ 已完成" if frontend else "⏳ 待開始"
        elif test_file is None:
            status = "⏳ 待開始"
        else:
            r = test_results.get(test_file)
            if r is None or not r.get("exists"):
                status = "⏳ 待開始"
            elif r["failed"] > 0 or r["errors"] > 0:
                status = "❌ 有問題"
            else:
                status = "✅ 已完成"

        label = f"  {phase_name:<12}{desc}"
        print(f"{label:<42}{status}")
    print()


# ═══════════════════════════════════════════════════════════════════
#  區塊 2: 測試結果明細
# ═══════════════════════════════════════════════════════════════════

def section_tests(test_results: dict[str, dict]) -> None:
    banner("🧪", "測試結果明細")
    print()

    all_test_files = sorted(set(
        tf for _, _, tf in PHASES if tf is not None
    ))

    for tf in all_test_files:
        r = test_results.get(tf)
        if r is None or not r.get("exists"):
            print(f"  {tf:<40}⏳ 尚未建立")
        elif r["failed"] > 0 or r["errors"] > 0:
            total_bad = r["failed"] + r["errors"]
            print(f"  {tf:<40}❌ {r['passed']} passed, {total_bad} failed")
            for f in r.get("failures", []):
                print(f"    └─ FAILED: {f}")
        else:
            print(f"  {tf:<40}✅ {r['passed']} passed")
    print()


# ═══════════════════════════════════════════════════════════════════
#  區塊 3: Knowledge Store 狀態
# ═══════════════════════════════════════════════════════════════════

def section_knowledge() -> None:
    banner("📚", "Knowledge Store 狀態")
    print()

    if not KNOWLEDGE_STORE_PATH.exists():
        print("  （尚無知識庫資料）")
        print()
        return

    try:
        data = json.loads(KNOWLEDGE_STORE_PATH.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("entries", [])

        total = len(entries)
        by_source: dict[str, int] = {}
        for e in entries:
            src = e.get("source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1

        print(f"  總條目數: {total}")
        if by_source:
            parts = [f"{src}: {cnt}" for src, cnt in sorted(by_source.items())]
            print(f"  按來源:   {' / '.join(parts)}")
        else:
            print("  按來源:   （無）")
    except Exception as exc:
        print(f"  ⚠️  讀取失敗: {exc}")
    print()


# ═══════════════════════════════════════════════════════════════════
#  區塊 4: Skill Graph 狀態
# ═══════════════════════════════════════════════════════════════════

def section_skill_graph() -> None:
    banner("🗺️", "Skill Graph 狀態")
    print()

    # Check module importability
    module_ok = False
    try:
        from skill_graph.skill_graph import SkillGraph
        _sg = SkillGraph()
        module_ok = True
        print("  模組匯入: ✅ 可用")
    except Exception as exc:
        print(f"  模組匯入: ❌ {exc}")

    # Check persistent data
    if SKILL_GRAPH_PATH.exists():
        try:
            data = json.loads(SKILL_GRAPH_PATH.read_text(encoding="utf-8"))
            n_nodes = data.get("num_skills", "?")
            n_edges = data.get("num_edges", "?")
            entropy = data.get("structural_entropy", "?")
            capacity = data.get("capacity", "?")
            print(f"  持久化資料: {SKILL_GRAPH_PATH.name}")
            print(f"    節點: {n_nodes}  邊: {n_edges}  entropy: {entropy}  capacity: {capacity}")

            # Tier breakdown
            nodes = data.get("nodes", [])
            tiers: dict[str, int] = {}
            for n in nodes:
                t = n.get("tier", "unknown")
                tiers[t] = tiers.get(t, 0) + 1
            if tiers:
                parts = [f"{t}: {c}" for t, c in sorted(tiers.items())]
                print(f"    Tier 分佈: {' / '.join(parts)}")
        except Exception as exc:
            print(f"  持久化資料: ⚠️ 讀取失敗 ({exc})")
    else:
        if module_ok:
            print("  持久化資料: （模組可用，尚無持久化資料）")
        else:
            print("  持久化資料: （模組不可用）")
    print()


# ═══════════════════════════════════════════════════════════════════
#  區塊 5: 程式碼統計
# ═══════════════════════════════════════════════════════════════════

def section_code_stats() -> None:
    banner("📁", "程式碼統計")
    print()

    total_files = 0
    total_lines = 0

    for category, patterns in CODE_CATEGORIES:
        files = resolve_globs(patterns)
        lines = sum(count_lines(f) for f in files)
        total_files += len(files)
        total_lines += lines
        print(f"  {category:<26}{len(files):>3} 檔案, {lines:>5} 行")

    print(f"  {'─' * 42}")
    print(f"  {'合計':<26}{total_files:>3} 檔案, {total_lines:>5} 行")
    print()


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    os.chdir(ROOT)

    print()
    print(f"{'═' * W}")
    print(f"  🔍 Self-Evolving Skill Graph — 專案檢查報告")
    print(f"{'═' * W}")

    # Run all tests first (collect results)
    test_files = [tf for _, _, tf in PHASES if tf is not None]
    test_results: dict[str, dict] = {}
    for tf in test_files:
        try:
            test_results[tf] = run_pytest(tf)
        except subprocess.TimeoutExpired:
            test_results[tf] = {
                "exists": True, "passed": 0, "failed": 0,
                "errors": 1, "failures": ["(timeout)"], "output": "",
            }
        except Exception as exc:
            test_results[tf] = {
                "exists": True, "passed": 0, "failed": 0,
                "errors": 1, "failures": [str(exc)], "output": "",
            }

    section_phases(test_results)
    section_tests(test_results)
    section_knowledge()
    section_skill_graph()
    section_code_stats()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  最後檢查時間: {now}")
    print()


if __name__ == "__main__":
    main()
