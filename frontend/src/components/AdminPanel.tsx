import { useState } from 'react';
import { ChevronDown, ChevronUp, MessageCircle, Send, SkipForward } from 'lucide-react';
import { useStore } from '@/hooks/useStore';
import { cn } from '@/lib/utils';

export function AdminPanel() {
    const pendingQueries = useStore((s) => s.pendingQueries);
    const respondToQuery = useStore((s) => s.respondToQuery);
    const skipQuery = useStore((s) => s.skipQuery);
    const [expanded, setExpanded] = useState(false);
    const [responses, setResponses] = useState<Record<string, string>>({});

    const count = pendingQueries.length;

    // Auto-expand when new queries arrive
    if (count > 0 && !expanded) {
        setExpanded(true);
    }

    if (count === 0 && !expanded) return null;

    return (
        <div className="glass border-t border-[hsl(var(--border))]">
            {/* Header bar */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-between px-5 py-2 text-xs hover:bg-[hsl(var(--accent))] transition-colors"
            >
                <div className="flex items-center gap-2">
                    <MessageCircle size={14} className="text-amber-400" />
                    <span className="font-medium">
                        Agent 有 {count} 個待回覆問題
                    </span>
                    {count > 0 && (
                        <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse-dot" />
                    )}
                </div>
                {expanded ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            </button>

            {/* Expanded panel */}
            {expanded && count > 0 && (
                <div className="px-5 pb-3 space-y-3 animate-slide-up">
                    {pendingQueries.map((q) => (
                        <div
                            key={q.query_id}
                            className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3"
                        >
                            <div className="flex items-start gap-2 mb-2">
                                <span className="text-amber-400 text-sm">🤖</span>
                                <p className="text-sm flex-1">{q.question}</p>
                                {/* 5-45: Context link → selectEpisode */}
                                {(q as any).episode_id !== undefined && (
                                    <button
                                        onClick={() => useStore.getState().selectEpisode((q as any).episode_id)}
                                        className="text-[9px] px-1.5 py-0.5 rounded bg-[hsl(var(--secondary))] text-blue-400 hover:text-blue-300 transition-colors"
                                        title="跳到這個 episode"
                                    >
                                        ep#{(q as any).episode_id}
                                    </button>
                                )}
                            </div>
                            <div className="flex items-end gap-2">
                                <input
                                    type="text"
                                    value={responses[q.query_id] || ''}
                                    onChange={(e) =>
                                        setResponses((r) => ({ ...r, [q.query_id]: e.target.value }))
                                    }
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') {
                                            const text = responses[q.query_id]?.trim();
                                            if (text) {
                                                respondToQuery(q.query_id, text);
                                                setResponses((r) => ({ ...r, [q.query_id]: '' }));
                                            }
                                        }
                                    }}
                                    placeholder="輸入回覆…"
                                    className={cn(
                                        'flex-1 rounded-md border border-[hsl(var(--border))]',
                                        'bg-[hsl(var(--secondary))] px-3 py-1.5 text-sm',
                                        'focus:outline-none focus:ring-1 focus:ring-amber-400',
                                    )}
                                />
                                <button
                                    onClick={() => {
                                        const text = responses[q.query_id]?.trim();
                                        if (text) {
                                            respondToQuery(q.query_id, text);
                                            setResponses((r) => ({ ...r, [q.query_id]: '' }));
                                        }
                                    }}
                                    className="p-1.5 rounded-md bg-amber-500 text-white hover:brightness-110 transition-all"
                                >
                                    <Send size={13} />
                                </button>
                                <button
                                    onClick={() => skipQuery(q.query_id)}
                                    className="p-1.5 rounded-md border border-[hsl(var(--border))] text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))] transition-all"
                                    title="跳過"
                                >
                                    <SkipForward size={13} />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
