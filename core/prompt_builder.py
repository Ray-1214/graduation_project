"""
Prompt template engine for all reasoning strategies.

Uses Python f-string style templates (no Jinja2 dependency).
Each template is keyed by strategy name and rendered via
PromptBuilder.build(name, **variables).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, str] = {

    # ── Chain of Thought (single-pass baseline) ──────────────────────────
    "cot": (
        "Solve the following problem step by step.\n\n"
        "Problem: {task}\n\n"
        "Let's think step by step:\n"
    ),

    # ── Tree of Thoughts: branch expansion ───────────────────────────────
    "tot_expand": (
        "Given the problem: {task}\n\n"
        "Current reasoning path:\n{current_path}\n\n"
        "Generate {n_branches} distinct next reasoning steps. "
        "Each step should explore a DIFFERENT approach or hypothesis.\n\n"
        "Format your response as:\n"
        "Step 1: [reasoning step]\n"
        "Step 2: [reasoning step]\n"
        "Step 3: [reasoning step]\n"
    ),

    # ── Tree of Thoughts: node evaluation ────────────────────────────────
    "tot_evaluate": (
        "Given the problem: {task}\n\n"
        "Evaluate the following reasoning path on a scale of 1-10.\n"
        "Consider correctness, completeness, and logical coherence.\n\n"
        "Reasoning path:\n{reasoning_path}\n\n"
        "Score (just the number): "
    ),

    # ── ReAct loop ───────────────────────────────────────────────────────
    "react": (
        "Answer the following question using the available tools.\n\n"
        "Question: {task}\n\n"
        "Available tools:\n{tool_descriptions}\n\n"
        "Strategy:\n"
        "- If you are unsure about the answer, first use web_search to look it up.\n"
        "- If web_search results are insufficient or unclear, use admin_query to ask the administrator.\n"
        "- Prefer using existing knowledge before making external calls.\n\n"
        "Use this EXACT format for each step:\n"
        "Thought: [your reasoning about what to do next]\n"
        "Action[tool_name]: [tool input]\n\n"
        "After a tool is used you will see:\n"
        "Observation: [tool result]\n\n"
        "When you have the final answer:\n"
        "Thought: I now have enough information.\n"
        "Finish[your final answer]\n\n"
        "{previous_steps}"
    ),

    # ── Reflexion: self-reflection writing ───────────────────────────────
    "reflexion": (
        "You completed a task. Reflect on your performance.\n\n"
        "Task: {task}\n\n"
        "Your trajectory:\n{trajectory}\n\n"
        "Outcome: {outcome}\n"
        "Score: {score}\n\n"
        "Write a concise self-reflection:\n"
        "1. What went well?\n"
        "2. What went wrong?\n"
        "3. What specific lessons should you remember for similar future tasks?\n\n"
        "Reflection:\n"
    ),

    # ── RAG: context injection ───────────────────────────────────────────
    "rag_context": (
        "Use the following retrieved context to help answer the question.\n"
        "If the context is not relevant, rely on your own knowledge.\n\n"
        "--- Retrieved Context ---\n"
        "{retrieved_context}\n"
        "--- End Context ---\n\n"
        "Question: {task}\n\n"
        "Answer:\n"
    ),

    # ── Evaluator: LLM-as-judge ──────────────────────────────────────────
    "evaluate": (
        "You are an impartial evaluator. Score the following answer "
        "to the given task on a scale from 0.0 to 1.0.\n\n"
        "Task: {task}\n\n"
        "Answer: {answer}\n\n"
        "Consider: correctness, completeness, clarity, and relevance.\n\n"
        "Score (a single decimal number between 0.0 and 1.0): "
    ),

    # ══════════════════════════════════════════════════════════════════
    #  CompoundReasoner templates
    # ══════════════════════════════════════════════════════════════════

    # ── Phase ① Structured Thinking: task analysis + plan ─────────────
    "structured_thinking": (
        "你是一個具有深度推理能力的 AI Agent。"
        "在開始執行任務之前，請先進行結構化分析：\n\n"
        "任務：{task}\n"
        "{available_skills}"
        "{past_lessons}"
        "\n請分析（必須嚴格遵守以下格式）：\n"
        "1. 任務類型：(事實查詢 / 推理計算 / 多步驟操作 / 創意生成)\n"
        "2. 預估複雜度：(simple / moderate / complex)\n"
        "3. 需要的工具：(web_search / calculator / file_ops / "
        "admin_query / none)（可多選，用逗號分隔）\n"
        "4. 推薦策略路線：(cot_only / react_cot / react_tot / "
        "full_compound)\n"
        "   - cot_only：簡單事實問答或單步推理\n"
        "   - react_cot：需要工具調用的任務\n"
        "   - react_tot：複雜分支推理 + 工具調用\n"
        "   - full_compound：最複雜任務（多工具 + 搜尋 + 推理 + 驗證）\n"
        "5. 子目標拆解：[step1, step2, ...]（2-5 個步驟）\n"
        "6. 風險與注意事項：（可能的陷阱或失敗模式）\n"
    ),

    # ── Phase ② Compound Thought: enhanced ReAct loop Thought ────────
    "compound_thought": (
        "你正在逐步解決一個任務。請深入思考這一步。\n\n"
        "當前任務：{task}\n"
        "計劃：{plan}\n"
        "已完成步驟：\n{completed_steps}\n"
        "當前子目標：{current_subgoal}\n\n"
        "請分析：\n"
        "- 目前的觀察結果和進展\n"
        "- 下一步行動是什麼？（需要搜尋？計算？直接推理？）\n"
        "- 如果有多條可能路線，列出各自的優缺點\n\n"
        "可用工具：\n{tool_descriptions}\n\n"
        "使用以下格式回覆：\n"
        "Thought: [你的深入推理]\n"
        "Action[tool_name]: [tool input]\n\n"
        "或者如果已有最終答案：\n"
        "Thought: 已收集足夠資訊。\n"
        "Finish[最終答案]\n"
    ),

    # ── Phase ② Self-Check: grounded mid-execution verification ──────
    "self_check": (
        "你正在執行一個多步驟任務，現在暫停進行自我檢查。\n\n"
        "任務：{task}\n"
        "原始計劃：{plan}\n"
        "已執行 {n_steps} 步：\n{steps_summary}\n\n"
        "自我檢查（請基於已有的工具結果和事實，而非憑感覺）：\n"
        "1. 目前進度是否符合預期？已完成哪些子目標？\n"
        "2. 最近的工具結果是否和之前的推理一致？有無矛盾？\n"
        "3. 方向是否正確，還是偏離了原始目標？\n"
        "4. 是否有遺漏或錯誤需要修正？\n"
        "5. 剩餘步驟的計劃是否需要修改？\n\n"
        "如果一切正常，回覆：「CONTINUE」\n"
        "如果需要調整，回覆：「REPLAN: {{新計劃描述}}」\n"
    ),

    # ── Phase ④ Strategy Reflection: post-task strategy analysis ──────
    "strategy_reflection": (
        "你剛完成了一個推理任務，請反思推理策略的選擇。\n\n"
        "任務：{task}\n"
        "使用的推理策略組合：{strategies_used}\n"
        "結果：{outcome}\n"
        "分數：{score}\n\n"
        "推理軌跡摘要：\n{trajectory_summary}\n\n"
        "請從三個層面反思：\n\n"
        "【策略教訓】（關於推理方法的選擇）\n"
        "- 策略選擇是否恰當？如果重做，會選不同策略嗎？\n"
        "- 各個推理步驟的效率如何？哪些是有效的？哪些是浪費的？\n\n"
        "【知識收穫】（關於這個問題領域學到的新資訊）\n"
        "- 在解決過程中發現了什麼新知識或關鍵事實？\n\n"
        "【錯誤警告】（要避免的陷阱）\n"
        "- 是否遇到了意外情況或陷阱？\n"
        "- 下次遇到類似任務，有什麼需要特別注意的？\n"
    ),

    # ── Think Tool: analyse observation without action ────────────────
    #    Ref: Anthropic "The think tool" (2025)
    #    τ-bench (arXiv:2406.12045): +54% with think tool + domain prompt
    "think_tool": (
        "[Think Step — 分析新資訊]\n"
        "剛收到的工具結果：\n{observation}\n\n"
        "目前任務：{task}\n"
        "當前子目標：{current_subgoal}\n"
        "已完成步驟：\n{completed_steps}\n\n"
        "請用這個空間進行深入分析（不需要產生行動）：\n"
        "1. 這個工具結果告訴我什麼？跟我預期的一致嗎？\n"
        "2. 有沒有意外的發現需要調整計劃？\n"
        "3. 這個資訊可靠嗎？是否和已知知識矛盾？\n"
        "4. 下一步最合理的行動是什麼？為什麼？\n\n"
        "注意：這是內部思考空間，不會直接呈現給使用者。\n"
    ),

    # ── Phase ③ Fact Check: hallucination detection & verification ────
    #    Ref: Huang et al. "LLMs Cannot Self-Correct Reasoning Yet"
    #    (ICLR 2024, arXiv:2310.01798)
    "fact_check": (
        "請檢查以下回答的事實準確性。\n\n"
        "問題：{task}\n"
        "回答：{answer}\n\n"
        "推理過程中使用的工具結果：\n{tool_results_summary}\n\n"
        "已知相關知識：\n{relevant_knowledge}\n\n"
        "逐項檢查：\n"
        "1. 回答中有哪些關鍵事實斷言？逐一列出。\n"
        "2. 每個斷言的驗證狀態：\n"
        "   - 有工具結果或已知知識支持 → [verified]\n"
        "   - 無直接支持但為合理推論 → [inferred]\n"
        "   - 與已知事實矛盾 → [conflict]\n"
        "   - 無法驗證 → [unverified]\n"
        "3. 如果有 [conflict] 或多個 [unverified]，建議修正方向。\n\n"
        "輸出格式：\n"
        "VERDICT: PASS / NEEDS_REVISION / UNRELIABLE\n"
        "CLAIMS:\n"
        "- claim: {{斷言內容}}, status: {{verified/inferred/"
        "conflict/unverified}}, evidence: {{支持證據}}\n"
        "REVISION: (如需修正，給出具體建議)\n"
    ),
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Renders prompt templates with variable substitution."""

    def __init__(self, extra_templates: Optional[Dict[str, str]] = None) -> None:
        self.templates: Dict[str, str] = {**TEMPLATES}
        if extra_templates:
            self.templates.update(extra_templates)

    def build(self, template_name: str, **variables: Any) -> str:
        """Render a named template with the given variables.

        Raises KeyError if the template name is unknown.
        Raises KeyError if a required variable is missing.
        """
        if template_name not in self.templates:
            raise KeyError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(self.templates.keys())}"
            )
        template = self.templates[template_name]
        try:
            return template.format(**variables)
        except KeyError as exc:
            raise KeyError(
                f"Missing variable {exc} for template '{template_name}'"
            ) from exc

    def register(self, name: str, template: str) -> None:
        """Register a new template (or override an existing one)."""
        self.templates[name] = template

    def list_templates(self) -> list[str]:
        """Return all registered template names."""
        return list(self.templates.keys())
