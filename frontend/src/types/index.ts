/* ── EpisodeDetail (matches backend schema) ────────────────── */

export interface TaskInfo {
    id: string;
    tier: number;
    description: string;
    expected_answer: string;
}

export interface ContextSlot {
    content: string;
    tokens: number;
    skills_injected?: string[];
    compression_tier?: number;
}

export interface ContextAssembly {
    slots: Record<string, ContextSlot>;
    total_tokens: number;
    budget: number;
    overflow_handled: boolean;
}

export interface ExecutionStep {
    step: number;
    type: string;
    content: string;
    tool?: string;
    tool_input?: unknown;
    tool_output?: string;
    duration_ms?: number;
    prompt_sent?: string;
    llm_response?: string;
}

export interface PhaseThinking {
    parsed_result: {
        content: string;
        problem_analysis?: string;
        strategy_choice?: string;
        key_considerations?: string[];
    };
    prompt_sent?: string;
    llm_response?: string;
    tokens_in?: number;
    tokens_out?: number;
    duration_ms: number;
}

export interface PhaseExecution {
    steps: ExecutionStep[];
    total_llm_calls: number;
    total_tool_calls: number;
    duration_ms: number;
}

export interface PhaseVerification {
    hallucination_guard: {
        overall_score: number;
        hallucination_detected: boolean;
        claims_extracted?: Array<{ claim: string; source: string }>;
        evidence_matching?: Array<{
            claim: string;
            evidence: string;
            verdict: string;
            confidence: number;
        }>;
    };
    self_check?: {
        confidence: number;
        issues_found: string[];
    };
    verdict: string;
    duration_ms: number;
}

export interface PhaseReflexion {
    reflection_text: string;
    reflexion_entries: Array<{
        type: string;
        content: string;
        routed_to?: string;
    }>;
    prompt_sent?: string;
    llm_response?: string;
    duration_ms: number;
}

export interface CompoundReasoning {
    strategy_selected: string;
    strategy_reason: string;
    phase_1_thinking: PhaseThinking;
    phase_2_execution: PhaseExecution;
    phase_3_verification: PhaseVerification;
    phase_4_reflexion: PhaseReflexion;
}

export interface EvolutionEvent {
    type: string;
    skill_id?: string;
    policy?: string;
    reason?: string;
}

export interface Evolution {
    events: EvolutionEvent[];
    graph_ops: {
        insertions: number;
        contractions: number;
        updates: number;
    };
    graph_after: {
        sigma_size: number;
        entropy: number;
    };
}

export interface MetricsSnapshot {
    rho: number;
    kappa: number;
    entropy: number;
    delta_sigma: number;
    planning_depth: number;
    sigma_size: number;
}

export interface EpisodeDetail {
    episode_id: number;
    task: TaskInfo;
    timestamp: string;
    duration_total_ms: number;
    context_assembly: ContextAssembly;
    compound_reasoning: CompoundReasoning;
    evolution: Evolution;
    metrics: MetricsSnapshot;
    answer: string;
    correct: boolean;
}

export interface EpisodeSummary {
    episode_id: number;
    task: string;
    strategy: string;
    correct: boolean;
    duration_ms: number;
    answer: string;
    timestamp: string;
}

/* ── Graph ─────────────────────────────────────────────────── */

export interface GraphNode {
    id: string;
    name?: string;
    policy?: string;
    utility?: number;
    tier?: string;
    frequency?: number;
    cost?: number;
    version?: number;
    created_episode?: number;
    // D3 simulation fields
    x?: number;
    y?: number;
    fx?: number | null;
    fy?: number | null;
    vx?: number;
    vy?: number;
}

export interface GraphEdge {
    source: string | GraphNode;
    target: string | GraphNode;
    src?: string;
    dst?: string;
    weight: number;
    type?: string;
}

export interface GraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
    entropy: number;
    capacity: number;
    sigma_size: number;
}

/* ── Metrics ───────────────────────────────────────────────── */

export interface MetricsRecord {
    episode_id: number;
    rho: number;
    kappa: number;
    entropy: number;
    delta_sigma: number;
    planning_depth: number;
    sigma_size: number;
    timestamp?: string;
}

export interface MetricsData {
    history: MetricsRecord[];
    iterations: Array<Record<string, unknown>>;
}

/* ── Admin ─────────────────────────────────────────────────── */

export interface PendingQuery {
    query_id: string;
    question: string;
    episode_id?: number;
    timestamp: string;
    context: string;
}

/* ── Command History ───────────────────────────────────────── */

export interface CommandEntry {
    command: string;
    mode: string;
    episode_id: number;
    timestamp: string;
    status: string;
}

/* ── WebSocket Events ──────────────────────────────────────── */

export interface WSEvent {
    type: string;
    [key: string]: unknown;
}
