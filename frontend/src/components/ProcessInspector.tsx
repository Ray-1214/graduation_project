import { useState } from 'react';
import {
    ChevronRight, CheckCircle2, XCircle, Clock,
    Wrench, Brain, Eye, Shield, Lightbulb, GitBranch,
    FileJson,
} from 'lucide-react';
import { useStore } from '@/hooks/useStore';
import { cn } from '@/lib/utils';
import type { EpisodeDetail, ExecutionStep } from '@/types';

/* ── Phase card config ────────────────────────────────────── */

const PHASES = [
    { key: 'context', label: '1. Context', icon: Brain, color: 'text-blue-400' },
    { key: 'thinking', label: '2. Thinking', icon: Lightbulb, color: 'text-yellow-400' },
    { key: 'execution', label: '3. Execution', icon: Wrench, color: 'text-green-400' },
    { key: 'verification', label: '4. Verification', icon: Shield, color: 'text-purple-400' },
    { key: 'reflexion', label: '5. Reflexion', icon: Eye, color: 'text-pink-400' },
    { key: 'evolution', label: '6. Evolution', icon: GitBranch, color: 'text-cyan-400' },
] as const;

const STEP_ICONS: Record<string, string> = {
    thought: '💭',
    action: '🔧',
    observation: '👁',
    finish: '✅',
    final_answer: '✅',
    error: '❌',
    evaluation: '📊',
    branch: '🌿',
    reflection: '🪞',
};

/* ── Sub-components ───────────────────────────────────────── */

function EpisodeList() {
    const episodes = useStore((s) => s.episodes);
    const selectedId = useStore((s) => s.selectedEpisodeId);
    const selectEpisode = useStore((s) => s.selectEpisode);

    return (
        <div className="w-full h-full overflow-y-auto">
            {episodes.length === 0 ? (
                <p className="text-xs text-[hsl(var(--muted-foreground))] p-3">
                    No episodes yet
                </p>
            ) : (
                episodes.map((ep) => (
                    <button
                        key={ep.episode_id}
                        onClick={() => selectEpisode(ep.episode_id)}
                        className={cn(
                            'w-full text-left px-3 py-2 border-b border-[hsl(var(--border))]',
                            'hover:bg-[hsl(var(--accent))] transition-colors text-xs',
                            selectedId === ep.episode_id && 'bg-[hsl(var(--accent))]',
                        )}
                    >
                        <div className="flex items-center gap-1.5 mb-0.5">
                            <span className="font-mono text-[10px] text-[hsl(var(--muted-foreground))]">
                                #{ep.episode_id}
                            </span>
                            {ep.correct ? (
                                <CheckCircle2 size={11} className="text-green-400" />
                            ) : (
                                <XCircle size={11} className="text-red-400" />
                            )}
                            <span className="px-1 py-0 rounded text-[9px] bg-[hsl(var(--secondary))] text-[hsl(var(--muted-foreground))]">
                                {ep.strategy}
                            </span>
                        </div>
                        <p className="truncate text-[hsl(var(--foreground))]">
                            {ep.task}
                        </p>
                        <div className="flex items-center gap-1 mt-0.5 text-[hsl(var(--muted-foreground))]">
                            <Clock size={9} />
                            <span className="text-[9px]">{(ep.duration_ms / 1000).toFixed(1)}s</span>
                        </div>
                    </button>
                ))
            )}
        </div>
    );
}

function PhaseCard({
    phase,
    episode,
    expanded,
    onToggle,
}: {
    phase: typeof PHASES[number];
    episode: EpisodeDetail;
    expanded: boolean;
    onToggle: () => void;
}) {
    const cr = episode.compound_reasoning;

    // Status badge
    let statusText = '—';
    let statusColor = 'text-[hsl(var(--muted-foreground))]';
    if (phase.key === 'context') {
        const tokens = episode.context_assembly.total_tokens;
        statusText = `${tokens}/${episode.context_assembly.budget} tok`;
        statusColor = 'text-blue-400';
    } else if (phase.key === 'thinking') {
        statusText = `${(cr.phase_1_thinking.duration_ms / 1000).toFixed(1)}s`;
        statusColor = 'text-yellow-400';
    } else if (phase.key === 'execution') {
        statusText = `${cr.phase_2_execution.total_llm_calls} LLM · ${cr.phase_2_execution.total_tool_calls} tool`;
        statusColor = 'text-green-400';
    } else if (phase.key === 'verification') {
        const v = cr.phase_3_verification;
        statusText = v.verdict === 'PASS' ? '✅ PASS' : v.verdict;
        statusColor = v.verdict === 'PASS' ? 'text-green-400' : 'text-red-400';
    } else if (phase.key === 'reflexion') {
        const count = cr.phase_4_reflexion.reflexion_entries.length;
        statusText = `${count} entries`;
        statusColor = 'text-pink-400';
    } else if (phase.key === 'evolution') {
        const ops = episode.evolution.graph_ops;
        statusText = `+${ops.insertions} / -${ops.contractions}`;
        statusColor = 'text-cyan-400';
    }

    return (
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--secondary))] overflow-hidden">
            <button
                onClick={onToggle}
                className="w-full flex items-center gap-2 px-3 py-2 hover:bg-[hsl(var(--accent))] transition-colors"
            >
                <ChevronRight
                    size={12}
                    className={cn('transition-transform', expanded && 'rotate-90')}
                />
                <phase.icon size={13} className={phase.color} />
                <span className="text-xs font-medium flex-1 text-left">{phase.label}</span>
                <span className={cn('text-[10px]', statusColor)}>{statusText}</span>
            </button>

            {expanded && (
                <div className="px-3 pb-3 animate-slide-up">
                    <PhaseContent phase={phase.key} episode={episode} />
                </div>
            )}
        </div>
    );
}

function PhaseContent({
    phase,
    episode,
}: {
    phase: string;
    episode: EpisodeDetail;
}) {
    const cr = episode.compound_reasoning;

    if (phase === 'context') {
        return (
            <div className="space-y-2">
                <div className="text-[10px] text-[hsl(var(--muted-foreground))]">
                    Token budget: {episode.context_assembly.total_tokens} / {episode.context_assembly.budget}
                </div>
                <div className="w-full h-1.5 rounded bg-[hsl(var(--accent))] overflow-hidden">
                    <div
                        className="h-full rounded bg-blue-500 transition-all"
                        style={{
                            width: `${Math.min(
                                100,
                                (episode.context_assembly.total_tokens / episode.context_assembly.budget) * 100,
                            )}%`,
                        }}
                    />
                </div>
                {Object.entries(episode.context_assembly.slots).map(([slot, data]) => (
                    <div key={slot} className="flex justify-between text-[10px]">
                        <span className="text-[hsl(var(--muted-foreground))]">{slot}</span>
                        <span>{data.tokens} tok</span>
                    </div>
                ))}
            </div>
        );
    }

    if (phase === 'thinking') {
        const th = cr.phase_1_thinking;
        return (
            <div className="space-y-2">
                <div className="text-xs">
                    Strategy: <span className="font-medium text-yellow-400">{cr.strategy_selected}</span>
                    {cr.strategy_reason && (
                        <span className="text-[hsl(var(--muted-foreground))]"> — {cr.strategy_reason}</span>
                    )}
                </div>
                {/* 5-24: Dual pane — prompt / response */}
                <div className="grid grid-cols-2 gap-2">
                    <div>
                        <div className="text-[9px] uppercase text-[hsl(var(--muted-foreground))] mb-1">Prompt Sent</div>
                        <pre className="text-[10px] bg-[hsl(var(--accent))] rounded p-2 whitespace-pre-wrap max-h-40 overflow-y-auto">
                            {(th as any).prompt_sent || '(not captured)'}
                        </pre>
                    </div>
                    <div>
                        <div className="text-[9px] uppercase text-[hsl(var(--muted-foreground))] mb-1">LLM Response</div>
                        <pre className="text-[10px] bg-[hsl(var(--accent))] rounded p-2 whitespace-pre-wrap max-h-40 overflow-y-auto">
                            {(th as any).llm_response || th.parsed_result.content || '(not captured)'}
                        </pre>
                    </div>
                </div>
            </div>
        );
    }

    if (phase === 'execution') {
        return (
            <div className="space-y-1.5">
                {cr.phase_2_execution.steps.map((step: ExecutionStep) => (
                    <div
                        key={step.step}
                        className="flex items-start gap-2 text-[11px] py-1 border-l-2 border-[hsl(var(--border))] pl-2"
                    >
                        <span>{STEP_ICONS[step.type] || '·'}</span>
                        <div className="flex-1">
                            <span className="text-[hsl(var(--muted-foreground))] text-[9px] uppercase mr-1">
                                {step.type}
                            </span>
                            {step.tool && (
                                <span className="text-green-400 font-mono text-[10px]">
                                    {step.tool}({typeof step.tool_input === 'string' ? step.tool_input : JSON.stringify(step.tool_input)})
                                </span>
                            )}
                            {step.tool_output && (
                                <span className="text-[hsl(var(--muted-foreground))]"> → {step.tool_output}</span>
                            )}
                            {!step.tool && step.content && (
                                <span className="text-[hsl(var(--foreground))]">{step.content}</span>
                            )}
                            {step.duration_ms !== undefined && step.duration_ms > 0 && (
                                <span className="ml-2 text-[9px] text-[hsl(var(--muted-foreground))]">
                                    {step.duration_ms}ms
                                </span>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        );
    }

    if (phase === 'verification') {
        const v = cr.phase_3_verification;
        const hg = v.hallucination_guard;
        const claims = (hg as any).claims || [];
        return (
            <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs">
                    <span>Self-check verdict:</span>
                    <span
                        className={cn(
                            'px-1.5 py-0.5 rounded text-[10px] font-medium',
                            v.verdict === 'PASS'
                                ? 'bg-green-500/10 text-green-400'
                                : v.verdict === 'not_run'
                                    ? 'bg-gray-500/10 text-gray-400'
                                    : 'bg-red-500/10 text-red-400',
                        )}
                    >
                        {v.verdict}
                    </span>
                    <span className="text-[10px] text-[hsl(var(--muted-foreground))]">
                        confidence: <span className={cn(
                            hg.hallucination_detected ? 'text-red-400' : 'text-green-400',
                        )}>{hg.overall_score.toFixed(2)}</span>
                    </span>
                </div>
                {/* 5-27: Claims / evidence table */}
                {claims.length > 0 && (
                    <div className="text-[10px]">
                        <div className="grid grid-cols-3 gap-1 font-semibold text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))] pb-1 mb-1">
                            <span>Claim</span><span>Evidence</span><span>Verdict</span>
                        </div>
                        {claims.map((c: any, i: number) => (
                            <div key={i} className="grid grid-cols-3 gap-1 py-0.5">
                                <span className="truncate">{c.claim}</span>
                                <span className="truncate text-[hsl(var(--muted-foreground))]">{c.evidence || '—'}</span>
                                <span className={cn(
                                    c.supported ? 'text-green-400' : 'text-red-400'
                                )}>{c.supported ? '✅ supported' : '❌ unsupported'}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    }

    if (phase === 'reflexion') {
        const r = cr.phase_4_reflexion;
        return (
            <div className="space-y-2">
                {r.reflection_text && (
                    <pre className="text-[10px] bg-[hsl(var(--accent))] rounded p-2 whitespace-pre-wrap max-h-40 overflow-y-auto">
                        {r.reflection_text}
                    </pre>
                )}
                {r.reflexion_entries.map((entry, i) => (
                    <div key={i} className="flex items-start gap-2 text-[10px]">
                        <span
                            className={cn(
                                'px-1 py-0.5 rounded text-[9px] font-medium',
                                entry.type?.includes('策略') && 'bg-blue-500/10 text-blue-400',
                                entry.type?.includes('知識') && 'bg-green-500/10 text-green-400',
                                entry.type?.includes('錯誤') && 'bg-red-500/10 text-red-400',
                            )}
                        >
                            {entry.type}
                        </span>
                        <span className="flex-1">{entry.content}</span>
                        {entry.routed_to && (
                            <span className="text-[hsl(var(--muted-foreground))]">
                                → {entry.routed_to}
                            </span>
                        )}
                    </div>
                ))}
            </div>
        );
    }

    if (phase === 'evolution') {
        const evo = episode.evolution;
        return (
            <div className="space-y-2">
                <div className="grid grid-cols-3 gap-2 text-[10px] text-center">
                    <div className="bg-[hsl(var(--accent))] rounded p-1.5">
                        <div className="text-cyan-400 font-semibold">{evo.graph_ops.insertions}</div>
                        <div className="text-[hsl(var(--muted-foreground))]">insert</div>
                    </div>
                    <div className="bg-[hsl(var(--accent))] rounded p-1.5">
                        <div className="text-amber-400 font-semibold">{evo.graph_ops.contractions}</div>
                        <div className="text-[hsl(var(--muted-foreground))]">contract</div>
                    </div>
                    <div className="bg-[hsl(var(--accent))] rounded p-1.5">
                        <div className="text-purple-400 font-semibold">{evo.graph_ops.updates}</div>
                        <div className="text-[hsl(var(--muted-foreground))]">update</div>
                    </div>
                </div>
                <div className="text-[10px] text-[hsl(var(--muted-foreground))]">
                    Graph: |Σ| = {evo.graph_after.sigma_size}, H = {evo.graph_after.entropy.toFixed(2)}
                </div>
                {evo.events.map((evt, i) => (
                    <div key={i} className="text-[10px] flex items-center gap-1">
                        <span>{evt.type.includes('inserted') ? '📥' : evt.type.includes('failed') ? '❌' : '🆕'}</span>
                        <span>{evt.type}</span>
                        {evt.skill_id && (
                            <span className="text-[hsl(var(--muted-foreground))] font-mono">
                                {evt.skill_id}
                            </span>
                        )}
                    </div>
                ))}
            </div>
        );
    }

    return null;
}

/* ── Process Inspector (main export) ──────────────────────── */

export function ProcessInspector() {
    const currentEpisode = useStore((s) => s.currentEpisode);
    const [expandedPhases, setExpandedPhases] = useState<Set<string>>(new Set());
    const [activeTab, setActiveTab] = useState<'pipeline' | 'json' | 'llm'>('pipeline');
    const [showList, setShowList] = useState(true);

    const togglePhase = (key: string) => {
        setExpandedPhases((prev) => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    return (
        <div className="w-full h-full flex overflow-hidden">
            {/* Episode list sidebar */}
            {showList && (
                <div className="w-44 shrink-0 border-r border-[hsl(var(--border))] overflow-hidden flex flex-col">
                    <div className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))] shrink-0">
                        Episodes
                    </div>
                    <div className="flex-1 overflow-y-auto">
                        <EpisodeList />
                    </div>
                </div>
            )}

            {/* Main area */}
            <div className="flex-1 flex flex-col overflow-hidden">
                {/* Tab bar */}
                <div className="flex items-center gap-0 px-2 py-1 border-b border-[hsl(var(--border))] shrink-0">
                    <button
                        onClick={() => setShowList(!showList)}
                        className="text-[10px] px-2 py-1 rounded text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))]"
                    >
                        {showList ? '◀' : '▶'} List
                    </button>
                    <div className="flex-1" />
                    {(['pipeline', 'json', 'llm'] as const).map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={cn(
                                'px-2 py-1 rounded text-[10px] transition-colors',
                                activeTab === tab
                                    ? 'bg-[hsl(var(--primary))] text-white'
                                    : 'text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))]',
                            )}
                        >
                            {tab === 'pipeline' && '🔬 Pipeline'}
                            {tab === 'json' && '📄 Raw JSON'}
                            {tab === 'llm' && '🤖 LLM Calls'}
                        </button>
                    ))}
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-3">
                    {!currentEpisode ? (
                        <div className="flex flex-col items-center justify-center h-full text-[hsl(var(--muted-foreground))]">
                            <FileJson size={32} className="mb-2 opacity-30" />
                            <p className="text-xs">Select an episode or run a task</p>
                        </div>
                    ) : activeTab === 'pipeline' ? (
                        <div className="space-y-2">
                            {/* Result banner */}
                            <div
                                className={cn(
                                    'rounded-lg px-3 py-2 flex items-center justify-between text-xs',
                                    currentEpisode.correct
                                        ? 'bg-green-500/10 border border-green-500/20'
                                        : 'bg-red-500/10 border border-red-500/20',
                                )}
                            >
                                <span>
                                    {currentEpisode.correct ? '✅' : '❌'}{' '}
                                    <span className="font-medium">{currentEpisode.answer}</span>
                                </span>
                                <span className="text-[hsl(var(--muted-foreground))]">
                                    {(currentEpisode.duration_total_ms / 1000).toFixed(1)}s ·{' '}
                                    {currentEpisode.compound_reasoning.strategy_selected}
                                </span>
                            </div>

                            {/* Phase pipeline */}
                            {PHASES.map((phase) => (
                                <PhaseCard
                                    key={phase.key}
                                    phase={phase}
                                    episode={currentEpisode}
                                    expanded={expandedPhases.has(phase.key)}
                                    onToggle={() => togglePhase(phase.key)}
                                />
                            ))}
                        </div>
                    ) : activeTab === 'json' ? (
                        <pre className="text-[10px] font-mono whitespace-pre-wrap bg-[hsl(var(--secondary))] rounded-lg p-3 max-h-full overflow-y-auto">
                            {JSON.stringify(currentEpisode, null, 2)}
                        </pre>
                    ) : (
                        <div className="space-y-2">
                            {/* 5-31: LLM Calls with phase, tokens, duration */}
                            <div className="grid grid-cols-5 gap-1 text-[9px] font-semibold uppercase text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))] pb-1">
                                <span>#</span><span>Phase</span><span>Tokens In</span><span>Tokens Out</span><span>Duration</span>
                            </div>
                            {currentEpisode.compound_reasoning.phase_2_execution.steps
                                .filter((s) => s.type === 'thought' || s.type === 'final_answer')
                                .map((step, i) => (
                                    <div
                                        key={i}
                                        className="rounded bg-[hsl(var(--secondary))] p-2 text-[10px]"
                                    >
                                        <div className="grid grid-cols-5 gap-1 text-[hsl(var(--muted-foreground))]">
                                            <span>{i + 1}</span>
                                            <span className="text-yellow-400">{(step as any).phase || step.type}</span>
                                            <span>{(step as any).tokens_in ?? '—'}</span>
                                            <span>{(step as any).tokens_out ?? '—'}</span>
                                            <span>{step.duration_ms ? `${step.duration_ms}ms` : '—'}</span>
                                        </div>
                                        <p className="mt-1 whitespace-pre-wrap">{step.content}</p>
                                    </div>
                                ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
