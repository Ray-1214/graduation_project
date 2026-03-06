import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { useStore } from '@/hooks/useStore';

const METRICS = [
    { key: 'rho', label: 'ρ (compression)', color: '#3b82f6', predict: '↑' },
    { key: 'kappa', label: 'κ (contraction)', color: '#a855f7', predict: '↓' },
    { key: 'entropy', label: 'H (entropy)', color: '#22c55e', predict: '→ ln(K)' },
    { key: 'delta_sigma', label: '|ΔΣ| (capacity)', color: '#f59e0b', predict: '→ 0' },
    { key: 'planning_depth', label: 'E[D] (depth)', color: '#ef4444', predict: '↓' },
] as const;

export function MetricsDashboard() {
    const metrics = useStore((s) => s.metrics);
    const history = metrics?.history || [];

    return (
        <div className="w-full h-full flex flex-col overflow-hidden">
            {/* Header */}
            <div className="px-3 py-2 flex items-center justify-between shrink-0">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]">
                    Metrics
                </span>
                <span className="text-[10px] text-[hsl(var(--muted-foreground))]">
                    {history.length} episodes
                </span>
            </div>

            {/* Charts */}
            <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-1">
                {history.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-xs text-[hsl(var(--muted-foreground))]">
                        No data yet — run tasks to see metrics
                    </div>
                ) : (
                    METRICS.map(({ key, label, color, predict }) => (
                        <div
                            key={key}
                            className="rounded-lg bg-[hsl(var(--secondary))] p-2"
                        >
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-[10px] font-medium">{label}</span>
                                <span className="text-[9px] px-1 py-0.5 rounded bg-[hsl(var(--accent))] text-[hsl(var(--muted-foreground))]">
                                    預測 {predict}
                                </span>
                            </div>
                            <ResponsiveContainer width="100%" height={60}>
                                <LineChart data={history}>
                                    <CartesianGrid
                                        strokeDasharray="3 3"
                                        stroke="hsl(222, 47%, 18%)"
                                    />
                                    <XAxis
                                        dataKey="episode_id"
                                        hide
                                    />
                                    <YAxis hide domain={['auto', 'auto']} />
                                    <Tooltip
                                        contentStyle={{
                                            background: 'hsl(222, 47%, 9%)',
                                            border: '1px solid hsl(222, 47%, 18%)',
                                            borderRadius: '6px',
                                            fontSize: '11px',
                                        }}
                                        labelFormatter={(v) => `Episode ${v}`}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey={key}
                                        stroke={color}
                                        strokeWidth={1.5}
                                        dot={false}
                                        activeDot={{ r: 3, fill: color }}
                                    />
                                    {key === 'delta_sigma' && (
                                        <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                                    )}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
