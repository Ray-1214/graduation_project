import { Activity, Zap, FlaskConical } from 'lucide-react';
import { useStore } from '@/hooks/useStore';
import { cn } from '@/lib/utils';

export function Header() {
    const episodes = useStore((s) => s.episodes);
    const mode = useStore((s) => s.mode);
    const setMode = useStore((s) => s.setMode);
    const isRunning = useStore((s) => s.isRunning);

    return (
        <header className="flex items-center justify-between px-5 py-3 glass border-b border-[hsl(var(--border))]">
            {/* Left — title */}
            <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <Zap size={16} className="text-white" />
                </div>
                <div>
                    <h1 className="text-sm font-semibold tracking-tight">
                        Self-Evolving Skill Graph
                    </h1>
                    <p className="text-xs text-[hsl(var(--muted-foreground))]">
                        Episode #{episodes.length} completed
                        {isRunning && (
                            <span className="ml-2 text-blue-400 animate-pulse-dot">
                                ● Running
                            </span>
                        )}
                    </p>
                </div>
            </div>

            {/* Center — mode tabs */}
            <div className="flex items-center bg-[hsl(var(--secondary))] rounded-lg p-0.5">
                <button
                    onClick={() => setMode('interactive')}
                    className={cn(
                        'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all',
                        mode === 'interactive'
                            ? 'bg-[hsl(var(--primary))] text-white shadow-sm'
                            : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]',
                    )}
                >
                    <Activity size={13} />
                    Interactive
                </button>
                <button
                    onClick={() => setMode('experiment')}
                    className={cn(
                        'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all',
                        mode === 'experiment'
                            ? 'bg-[hsl(var(--primary))] text-white shadow-sm'
                            : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]',
                    )}
                >
                    <FlaskConical size={13} />
                    Experiment
                </button>
            </div>

            {/* Right — decorative status */}
            <div className="flex items-center gap-2 text-xs text-[hsl(var(--muted-foreground))]">
                <span>Skills: {episodes.length > 0 ? '—' : '0'}</span>
            </div>
        </header>
    );
}
