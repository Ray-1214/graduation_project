import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import { useStore } from '@/hooks/useStore';
import { cn } from '@/lib/utils';

export function TimelineSlider() {
    const episodes = useStore((s) => s.episodes);
    const selectedId = useStore((s) => s.selectedEpisodeId);
    const selectEpisode = useStore((s) => s.selectEpisode);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(1);
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const total = episodes.length;
    const current = selectedId ?? 0;

    // Playback
    useEffect(() => {
        if (isPlaying && total > 0) {
            intervalRef.current = setInterval(() => {
                const s = useStore.getState();
                const next = (s.selectedEpisodeId ?? -1) + 1;
                if (next >= s.episodes.length) {
                    setIsPlaying(false);
                } else {
                    s.selectEpisode(next);
                }
            }, 2000 / speed);
        }
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [isPlaying, speed, total]);

    if (total === 0) {
        return (
            <div className="px-5 py-2 glass border-t border-[hsl(var(--border))] text-center text-[10px] text-[hsl(var(--muted-foreground))]">
                No episodes — run a task to start
            </div>
        );
    }

    return (
        <div className="flex items-center gap-3 px-5 py-2 glass border-t border-[hsl(var(--border))]">
            {/* Controls */}
            <div className="flex items-center gap-1">
                <button
                    onClick={() => current > 0 && selectEpisode(current - 1)}
                    disabled={current <= 0}
                    className="p-1 rounded hover:bg-[hsl(var(--accent))] disabled:opacity-30 transition-colors"
                >
                    <SkipBack size={13} />
                </button>
                <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="p-1.5 rounded-full bg-[hsl(var(--primary))] text-white hover:brightness-110 transition-all"
                >
                    {isPlaying ? <Pause size={13} /> : <Play size={13} />}
                </button>
                <button
                    onClick={() => current < total - 1 && selectEpisode(current + 1)}
                    disabled={current >= total - 1}
                    className="p-1 rounded hover:bg-[hsl(var(--accent))] disabled:opacity-30 transition-colors"
                >
                    <SkipForward size={13} />
                </button>
            </div>

            {/* Slider */}
            <input
                type="range"
                min={0}
                max={Math.max(0, total - 1)}
                value={current}
                onChange={(e) => selectEpisode(Number(e.target.value))}
                className="flex-1 h-1 accent-[hsl(var(--primary))] cursor-pointer"
            />

            {/* Info */}
            <span className="text-xs text-[hsl(var(--muted-foreground))] min-w-[100px] text-right">
                Episode {current + 1} / {total}
            </span>

            {/* Speed */}
            <div className="flex items-center gap-1">
                {[0.5, 1, 2, 5].map((s) => (
                    <button
                        key={s}
                        onClick={() => setSpeed(s)}
                        className={cn(
                            'text-[9px] px-1.5 py-0.5 rounded transition-colors',
                            speed === s
                                ? 'bg-[hsl(var(--primary))] text-white'
                                : 'text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))]',
                        )}
                    >
                        {s}x
                    </button>
                ))}
            </div>
        </div>
    );
}
