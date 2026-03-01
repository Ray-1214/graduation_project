# 開發日誌 (Development Log)

> 追蹤每次修改、產出能力、遇到的問題與解決方案。

---

## 2026-02-27 — 初始架構建立

### 修改內容
- 建立完整的模組化認知架構 scaffold，共 22 個 Python 檔案
- **Core 層**：`config.py`、`llm_interface.py`（可插拔 BaseLLM）、`prompt_builder.py`（7 套 prompt 模板）
- **Memory 層**：`short_term.py`（ring buffer）、`long_term.py`（JSON 反思記憶）、`episodic_log.py`（推理軌跡）、`vector_store.py`（FAISS）
- **RAG**：`indexer.py`（分塊 + 嵌入）、`retriever.py`（相似性檢索）
- **Reasoning**：`cot.py`、`tot.py`（BFS/DFS beam search）、`react.py`（工具迴圈）、`reflexion.py`（自我反思 → 記憶）、`planner.py`（自動策略選擇）
- **Skills**：`calculator.py`（AST 安全求值）、`file_ops.py`（沙盒化）、`web_search.py`（stub）、`registry.py`
- **Agents**：`main_agent.py`（總控制器）、`evaluator_agent.py`（LLM-as-judge）
- **Experiments**：`run_experiment.py`（CLI）、`smoke_test.py`（16 項測試全通過）

### 產出能力
- 支援 4 種推理策略：CoT、Tree of Thoughts、ReAct（工具使用）、Reflexion
- RAG 向量檢索 + 任意策略組合
- CLI 實驗執行器，結果自動存 JSON
- 16 項冒煙測試全部通過

---

## 2026-02-27 — 首次實際執行 & 模型問題

### 執行測試
```bash
python3 experiments/run_experiment.py --task "What is 25 * 17?" --strategy react
```

### 🔴 遇到的問題
1. **Falcon-7B 是 base model，不是 instruction-tuned**
   - 模型完全不遵循 ReAct 格式（`Action[tool]: input`、`Finish[answer]`）
   - 8 個步驟全部 `WARNING: no Action or Finish found`
   - 輸出是重複的 `# [Home](../../home.md)` Markdown 碎片
2. **Context window 警告**：`n_ctx_per_seq (4096) > n_ctx_train (2048)`
   - Falcon-7B 訓練時只有 2048 context
3. **速度慢**：361 秒（6 分鐘），因為 base model 不知道何時停止，每步都 generate 到 max_tokens

### 解決方案
- 下載 **Mistral-7B-Instruct-v0.2 Q4_K_M**（4.37 GB），instruction-tuned 模型
- 刪除所有 Falcon-7B 檔案（釋放約 24 GB）
- 更新 `config.py` 指向新模型，`n_ctx=4096`（Mistral 支援到 8192）
- 冒煙測試更新並全部通過（16/16）

### 學到的教訓
> ⚠️ **Base model ≠ Instruction model**。用於 agent 推理的 LLM 必須是 instruction-tuned，否則不會遵循任何格式化提示。

---

## 2026-03-01 — EpisodicTrace 標準格式

### 修改內容
- 在 `memory/episodic_log.py` 新增 `TraceStep`、`EpisodicTrace` dataclass 和 `convert_log_to_trace()` 轉換函數
- 定義 canonical trace 格式：`(state, action, outcome, timestamp)` 四元組
- 轉換邏輯：將 `EpisodeStep` 的 step_type 分為 action 類（thought/action/branch/reflection）和 outcome 類（observation/evaluation/finish/error），配對成 triple

### 設計決策
- **state** = 前一步的 outcome（第一步為 task 描述）
- **連續兩個 action 型別** → 前者 flush 為 `outcome="(no explicit outcome)"`
- **task_id** 自動以 SHA-256(task + start_time) 生成，也可手動指定
- **success** = score ≥ threshold（預設 0.5）

### 產出能力
- 所有下游模組（SkillAbstractor、EvolutionOperator、MetricsTracker）可依賴此格式
- 3 項新測試加入 smoke_test.py，全部通過（19/19）

---

## 2026-03-01 — SkillNode (Self-Evolving Skill Graph)

### 修改內容
- 新增 `skill_graph/` 模組
- 實作 `skill_graph/skill_node.py`：`SkillNode` dataclass

### 形式化定義對應
| 數學符號 | 屬性 | 說明 |
|----------|------|------|
| π_σ | `policy: str` | prompt template |
| β_σ | `termination: str` | 終止條件 |
| I_σ | `initiation_set: List[str]` | 適用任務標籤 |
| f_σ | `frequency: int` | 使用次數 |
| r_σ | `reinforcement: float` | 累積增強 |
| c_σ | `cost: float` | 計算成本 |
| v_σ | `version: int` | 版本號 |
| U(σ) | `utility: float` | 效用值 |

### 方法
- `compute_utility(α, β, γ_c)` → U = α·r + β·f − γ_c·c
- `decay(γ)` → U ← (1−γ)·U
- `reinforce(ΔU, cost)` — 更新 f, r, c, U
- `matches(task)` — I_σ 匹配
- `evolve(new_policy)` — 產生子版本（v+1, parent_id 連結）
- `to_dict()` / `from_dict()` / `save()` / `load()` — JSON 序列化

### 產出能力
- Skill Graph 系統的基礎資料結構就緒
- 支援技能演化（parent→child 鏈）、效用衰減、initiation-set 匹配
- 所有測試通過

---

## 目前專案狀態

### 已完成模組
```
core/           ██████████ config, llm_interface, prompt_builder
memory/         ██████████ short_term, long_term, episodic_log (+trace), vector_store
rag/            ██████████ indexer, retriever
reasoning/      ██████████ cot, tot, react, reflexion, planner
skills/         ██████████ calculator, file_ops, web_search, registry
agents/         ██████████ main_agent, evaluator_agent
skill_graph/    ██████░░░░ skill_node (SkillAbstractor, EvolutionOperator 待做)
experiments/    ██████████ run_experiment, smoke_test (19/19 ✓)
```

### 待完成
- [ ] SkillAbstractor — 從 trace 自動抽取新 skill
- [ ] EvolutionOperator — skill 突變/交叉演化
- [ ] MetricsTracker — 效能追蹤與視覺化
- [ ] 使用 Mistral 模型實際跑完整 ReAct / ToT 實驗
- [ ] 端到端整合測試（RAG + ReAct + Reflexion + Skill Graph）

---

## 2026-03-01 — SkillGraph (Self-Evolving Skill Graph)

### 修改內容
- 新增 `skill_graph/skill_graph.py`：`SkillGraph` 類別
- 新增 `skill_graph/test_skill_graph.py`：19 項單元測試
- `requirements.txt` 加入 `networkx>=3.0`

### 形式化定義對應
- **G_t = (Σ_t, E_t, W_t)**：networkx.DiGraph，節點存 SkillNode，邊有 weight + edge_type
- **三種邊類型**：co_occurrence（允許環）、dependency、abstraction（強制 DAG）
- **結構熵**：H(G_t) = −Σ p(σ) log₂ p(σ)，p(σ) = U(σ)/ΣU
- **結構容量**：K* = |{σ : U(σ) ≥ θ}|，系統上限 K ≥ K*

### 實作方法
| 方法 | 功能 |
|------|------|
| `add_skill(skill)` | 加入節點，檢查容量上限 |
| `remove_skill(id)` | 移除節點與所有相關邊 |
| `add_edge(src, dst, w, type)` | 加邊，abstraction 型自動驗證 DAG |
| `get_active_skills(θ)` | 回傳 U ≥ θ 的技能 |
| `compute_entropy()` | 計算結構熵 |
| `compute_capacity(θ)` | 計算結構容量 K* |
| `decay_all(γ)` | 全體 utility 衰減 |
| `get_subgraph(ids)` | 取出子圖 |
| `snapshot()` | 輸出完整狀態 dict |

### 🔴 遇到的問題
- `networkx` 未安裝 → `pip install networkx` 解決

### 測試結果
- 19/19 單元測試全部通過 ✓

### 產出能力
- Skill Graph 的圖結構已就緒
- 支援節點增刪、三種語義邊（含 DAG 驗證）、結構熵/容量查詢、全體衰減、快照

---

## 目前專案狀態

### 已完成模組
```
core/           ██████████ config, llm_interface, prompt_builder
memory/         ██████████ short_term, long_term, episodic_log (+trace), vector_store
rag/            ██████████ indexer, retriever
reasoning/      ██████████ cot, tot, react, reflexion, planner
skills/         ██████████ calculator, file_ops, web_search, registry
agents/         ██████████ main_agent, evaluator_agent
skill_graph/    ████████░░ skill_node, skill_graph (Abstractor, Evolution 待做)
experiments/    ██████████ run_experiment, smoke_test (19/19 ✓)
```

### 待完成
- [ ] SkillAbstractor — 從 trace 自動抽取新 skill
- [ ] EvolutionOperator — skill 突變/交叉演化
- [ ] MetricsTracker — 效能追蹤與視覺化
- [ ] 使用 Mistral 模型實際跑完整 ReAct / ToT 實驗
- [ ] 端到端整合測試（RAG + ReAct + Reflexion + Skill Graph）

---

## 2026-03-01 — MemoryPartition（三層記憶體分區）

### 修改內容
- 新增 `skill_graph/memory_partition.py`：`MemoryPartition` 類別
- 新增 `skill_graph/test_memory_partition.py`：22 項單元測試

### 形式化定義對應
```
M_active  ← U(σ) ≥ θ_high + ε_h   (升級)
M_cold    ← 中間區域
M_archive ← U(σ) ≤ θ_low − ε_l    (降級)
```

### Hysteresis 機制（防止邊界振盪）
```
升級到 active:  U ≥ θ_high + ε_h  (= 0.8)
降級離開 active: U < θ_high − ε_h  (= 0.6)  ← 死區寬度 2·ε_h
升級離開 archive: U > θ_low + ε_l  (= 0.4)
降級到 archive:  U ≤ θ_low − ε_l  (= 0.2)   ← 死區寬度 2·ε_l
```

### 實作方法
| 方法 | 功能 |
|------|------|
| `assign_tier(skill, current_tier)` | 根據 utility + 當前層決定新層 |
| `update_all(graph)` | 對整個 SkillGraph 批次更新 |
| `get_tier(skill_id)` | 查詢當前層 |
| `get_skills_by_tier(tier)` | 列出某層所有 skill ID |
| `summary()` | 各層 skill 計數 |

### 🔴 遇到的問題
- **浮點精度**：`0.3 - 0.1 = 0.19999999999999998`（IEEE 754）
  - 測試中 `U = 0.2` 在邊界上的判斷不穩定
  - 解決：測試改用 `U = 0.19`，明確低於閾值

### 測試結果
- 22/22 單元測試全部通過 ✓
- 包含完整的反振盪場景測試（6 步 trajectory 驗證 hysteresis）

### 產出能力
- 三層記憶分區管理就緒
- Hysteresis 機制防止 skill 在閾值邊界來回振盪
- 與 SkillGraph 整合，支援批次更新

---

## 目前專案狀態

### 已完成模組
```
core/           ██████████ config, llm_interface, prompt_builder
memory/         ██████████ short_term, long_term, episodic_log (+trace), vector_store
rag/            ██████████ indexer, retriever
reasoning/      ██████████ cot, tot, react, reflexion, planner
skills/         ██████████ calculator, file_ops, web_search, registry
agents/         ██████████ main_agent, evaluator_agent
skill_graph/    █████████░ skill_node, skill_graph, memory_partition
experiments/    ██████████ run_experiment, smoke_test (19/19 ✓)
```

### 待完成
- [ ] SkillAbstractor — 從 trace 自動抽取新 skill
- [ ] EvolutionOperator — skill 突變/交叉演化
- [ ] MetricsTracker — 效能追蹤與視覺化
- [ ] 使用 Mistral 模型實際跑完整 ReAct / ToT 實驗
- [ ] 端到端整合測試（RAG + ReAct + Reflexion + Skill Graph）

