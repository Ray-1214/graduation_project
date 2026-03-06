/**
 * Zustand global store — shared state across all panels.
 */

import { create } from 'zustand';
import type {
    EpisodeDetail,
    EpisodeSummary,
    GraphData,
    MetricsData,
    PendingQuery,
    CommandEntry,
    WSEvent,
} from '@/types';
import { apiGet, apiPost } from '@/lib/api';

interface AppState {
    /* ── Data ─────────────────────────────────────────────────── */
    episodes: EpisodeSummary[];
    currentEpisode: EpisodeDetail | null;
    graph: GraphData | null;
    metrics: MetricsData | null;
    pendingQueries: PendingQuery[];
    commandHistory: CommandEntry[];
    wsEvents: WSEvent[];

    /* ── UI ──────────────────────────────────────────────────── */
    selectedEpisodeId: number | null;
    isRunning: boolean;
    mode: 'interactive' | 'experiment';

    /* ── Actions ─────────────────────────────────────────────── */
    fetchEpisodes: () => Promise<void>;
    fetchEpisode: (id: number) => Promise<void>;
    fetchGraph: () => Promise<void>;
    fetchMetrics: () => Promise<void>;
    fetchPending: () => Promise<void>;
    fetchHistory: () => Promise<void>;
    runTask: (desc: string, mode: string, expected?: string) => Promise<EpisodeDetail | null>;
    respondToQuery: (queryId: string, response: string) => Promise<void>;
    skipQuery: (queryId: string) => Promise<void>;
    addWSEvent: (event: WSEvent) => void;
    selectEpisode: (id: number) => void;
    setMode: (mode: 'interactive' | 'experiment') => void;
}

export const useStore = create<AppState>((set, get) => ({
    episodes: [],
    currentEpisode: null,
    graph: null,
    metrics: null,
    pendingQueries: [],
    commandHistory: [],
    wsEvents: [],
    selectedEpisodeId: null,
    isRunning: false,
    mode: 'interactive',

    fetchEpisodes: async () => {
        try {
            const data = await apiGet<EpisodeSummary[]>('/api/episodes');
            set({ episodes: data });
        } catch { /* ignore */ }
    },

    fetchEpisode: async (id: number) => {
        try {
            const data = await apiGet<EpisodeDetail>(`/api/episode/${id}`);
            if (!('error' in data)) {
                set({ currentEpisode: data, selectedEpisodeId: id });
            }
        } catch { /* ignore */ }
    },

    fetchGraph: async () => {
        try {
            const data = await apiGet<GraphData>('/api/graph');
            set({ graph: data });
        } catch { /* ignore */ }
    },

    fetchMetrics: async () => {
        try {
            const data = await apiGet<MetricsData>('/api/metrics');
            set({ metrics: data });
        } catch { /* ignore */ }
    },

    fetchPending: async () => {
        try {
            const data = await apiGet<PendingQuery[]>('/api/admin/pending');
            set({ pendingQueries: data });
        } catch { /* ignore */ }
    },

    fetchHistory: async () => {
        try {
            const data = await apiGet<CommandEntry[]>('/api/history');
            set({ commandHistory: data });
        } catch { /* ignore */ }
    },

    runTask: async (desc, mode, expected = '') => {
        set({ isRunning: true });
        try {
            const data = await apiPost<EpisodeDetail>('/api/run', {
                task_description: desc,
                mode,
                expected_answer: expected,
            });
            if ('error' in data) {
                set({ isRunning: false });
                return null;
            }
            // Refresh all panels
            const s = get();
            await Promise.all([
                s.fetchEpisodes(),
                s.fetchGraph(),
                s.fetchMetrics(),
                s.fetchHistory(),
            ]);
            set({
                currentEpisode: data,
                selectedEpisodeId: data.episode_id,
                isRunning: false,
            });
            return data;
        } catch {
            set({ isRunning: false });
            return null;
        }
    },

    respondToQuery: async (queryId, response) => {
        await apiPost('/api/admin/respond', { query_id: queryId, response });
        get().fetchPending();
    },

    skipQuery: async (queryId) => {
        await apiPost('/api/admin/skip', { query_id: queryId });
        get().fetchPending();
    },

    addWSEvent: (event) => {
        set((s) => ({ wsEvents: [...s.wsEvents.slice(-200), event] }));
        // Auto-refresh on episode_complete
        if (event.type === 'episode_complete') {
            const s = get();
            s.fetchEpisodes();
            s.fetchGraph();
            s.fetchMetrics();
            s.fetchPending();
        }
        if (event.type === 'admin_query') {
            get().fetchPending();
        }
    },

    selectEpisode: (id) => {
        set({ selectedEpisodeId: id });
        get().fetchEpisode(id);
    },

    setMode: (mode) => set({ mode }),
}));
