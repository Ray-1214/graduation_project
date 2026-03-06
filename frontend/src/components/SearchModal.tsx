import { useState, useEffect, useCallback } from 'react';
import { Search, X } from 'lucide-react';
import { useStore } from '@/hooks/useStore';
import { cn } from '@/lib/utils';

/**
 * 5-33: Ctrl+K global search modal with episode filtering.
 */
export function SearchModal() {
    const [open, setOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [filterStrategy, setFilterStrategy] = useState('');
    const [filterCorrect, setFilterCorrect] = useState<'' | 'true' | 'false'>('');

    const episodes = useStore((s) => s.episodes);
    const selectEpisode = useStore((s) => s.selectEpisode);

    // Ctrl+K listener
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                setOpen((v) => !v);
            }
            if (e.key === 'Escape') setOpen(false);
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);

    const filtered = episodes.filter((ep) => {
        if (query && !ep.task.toLowerCase().includes(query.toLowerCase())) return false;
        if (filterStrategy && ep.strategy !== filterStrategy) return false;
        if (filterCorrect === 'true' && !ep.correct) return false;
        if (filterCorrect === 'false' && ep.correct) return false;
        return true;
    });

    const strategies = [...new Set(episodes.map((ep) => ep.strategy))];

    const handleSelect = useCallback(
        (id: number) => {
            selectEpisode(id);
            setOpen(false);
        },
        [selectEpisode],
    );

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]" onClick={() => setOpen(false)}>
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
            <div
                className="relative w-[560px] max-h-[60vh] bg-[hsl(var(--card))] border border-[hsl(var(--border))] rounded-xl shadow-2xl flex flex-col overflow-hidden animate-slide-up"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Search input */}
                <div className="flex items-center gap-2 px-4 py-3 border-b border-[hsl(var(--border))]">
                    <Search size={16} className="text-[hsl(var(--muted-foreground))]" />
                    <input
                        autoFocus
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search episodes..."
                        className="flex-1 bg-transparent outline-none text-sm text-[hsl(var(--foreground))] placeholder:text-[hsl(var(--muted-foreground))]"
                    />
                    <button onClick={() => setOpen(false)} className="p-1 rounded hover:bg-[hsl(var(--accent))]">
                        <X size={14} />
                    </button>
                </div>

                {/* Filters */}
                <div className="flex items-center gap-2 px-4 py-2 text-[10px] text-[hsl(var(--muted-foreground))] border-b border-[hsl(var(--border))]">
                    <span>Filter:</span>
                    <select
                        value={filterStrategy}
                        onChange={(e) => setFilterStrategy(e.target.value)}
                        className="bg-[hsl(var(--secondary))] rounded px-1.5 py-0.5 text-[10px] border border-[hsl(var(--border))]"
                    >
                        <option value="">All strategies</option>
                        {strategies.map((s) => (
                            <option key={s} value={s}>{s}</option>
                        ))}
                    </select>
                    <select
                        value={filterCorrect}
                        onChange={(e) => setFilterCorrect(e.target.value as '' | 'true' | 'false')}
                        className="bg-[hsl(var(--secondary))] rounded px-1.5 py-0.5 text-[10px] border border-[hsl(var(--border))]"
                    >
                        <option value="">All results</option>
                        <option value="true">✅ Correct</option>
                        <option value="false">❌ Incorrect</option>
                    </select>
                    <span className="ml-auto">{filtered.length} / {episodes.length}</span>
                </div>

                {/* Results */}
                <div className="flex-1 overflow-y-auto">
                    {filtered.length === 0 ? (
                        <p className="text-center text-xs text-[hsl(var(--muted-foreground))] py-8">No matches</p>
                    ) : (
                        filtered.map((ep) => (
                            <button
                                key={ep.episode_id}
                                onClick={() => handleSelect(ep.episode_id)}
                                className="w-full text-left px-4 py-2 hover:bg-[hsl(var(--accent))] transition-colors border-b border-[hsl(var(--border))] last:border-b-0"
                            >
                                <div className="flex items-center gap-2 text-xs">
                                    <span className="font-mono text-[10px] text-[hsl(var(--muted-foreground))]">
                                        #{ep.episode_id}
                                    </span>
                                    <span className={cn(
                                        'text-[9px] px-1 py-0.5 rounded',
                                        ep.correct ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400',
                                    )}>
                                        {ep.correct ? '✅' : '❌'}
                                    </span>
                                    <span className="text-[9px] px-1 py-0.5 rounded bg-[hsl(var(--secondary))] text-[hsl(var(--muted-foreground))]">
                                        {ep.strategy}
                                    </span>
                                    <span className="flex-1 truncate">{ep.task}</span>
                                    <span className="text-[9px] text-[hsl(var(--muted-foreground))]">
                                        {(ep.duration_ms / 1000).toFixed(1)}s
                                    </span>
                                </div>
                            </button>
                        ))
                    )}
                </div>

                {/* Keyboard hint */}
                <div className="px-4 py-1.5 border-t border-[hsl(var(--border))] text-[9px] text-[hsl(var(--muted-foreground))] flex items-center gap-3">
                    <span><kbd className="px-1 py-0.5 rounded bg-[hsl(var(--secondary))]">↑↓</kbd> navigate</span>
                    <span><kbd className="px-1 py-0.5 rounded bg-[hsl(var(--secondary))]">Enter</kbd> select</span>
                    <span><kbd className="px-1 py-0.5 rounded bg-[hsl(var(--secondary))]">Esc</kbd> close</span>
                </div>
            </div>
        </div>
    );
}
