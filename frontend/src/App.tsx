import { useEffect } from 'react';
import { Header } from '@/components/Header';
import { CommandBar } from '@/components/CommandBar';
import { AdminPanel } from '@/components/AdminPanel';
import { GraphView } from '@/components/GraphView';
import { MetricsDashboard } from '@/components/MetricsDashboard';
import { ProcessInspector } from '@/components/ProcessInspector';
import { TimelineSlider } from '@/components/TimelineSlider';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useStore } from '@/hooks/useStore';

export default function App() {
  const mode = useStore((s) => s.mode);
  const fetchGraph = useStore((s) => s.fetchGraph);
  const fetchEpisodes = useStore((s) => s.fetchEpisodes);
  const fetchMetrics = useStore((s) => s.fetchMetrics);
  const fetchPending = useStore((s) => s.fetchPending);
  const fetchHistory = useStore((s) => s.fetchHistory);

  // Connect WebSocket
  useWebSocket();

  // Initial data load
  useEffect(() => {
    fetchGraph();
    fetchEpisodes();
    fetchMetrics();
    fetchPending();
    fetchHistory();
  }, [fetchGraph, fetchEpisodes, fetchMetrics, fetchPending, fetchHistory]);

  return (
    <div className="h-screen flex flex-col bg-[hsl(var(--background))] text-[hsl(var(--foreground))]">
      {/* Header */}
      <Header />

      {/* Command Bar (interactive mode) */}
      {mode === 'interactive' && <CommandBar />}

      {/* Three-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Panel 1: Graph View (35%) */}
        <div className="w-[35%] border-r border-[hsl(var(--border))] overflow-hidden">
          <GraphView />
        </div>

        {/* Panel 2: Metrics Dashboard (25%) */}
        <div className="w-[25%] border-r border-[hsl(var(--border))] overflow-hidden">
          <MetricsDashboard />
        </div>

        {/* Panel 3: Process Inspector (40%) */}
        <div className="w-[40%] overflow-hidden">
          <ProcessInspector />
        </div>
      </div>

      {/* Admin Response Panel */}
      <AdminPanel />

      {/* Timeline Slider */}
      <TimelineSlider />
    </div>
  );
}
