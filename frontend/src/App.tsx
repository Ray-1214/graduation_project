import { useEffect, useCallback } from 'react';
import { Header } from '@/components/Header';
import { CommandBar } from '@/components/CommandBar';
import { AdminPanel } from '@/components/AdminPanel';
import { GraphView } from '@/components/GraphView';
import { MetricsDashboard } from '@/components/MetricsDashboard';
import { ProcessInspector } from '@/components/ProcessInspector';
import { TimelineSlider } from '@/components/TimelineSlider';
import { SearchModal } from '@/components/SearchModal';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useStore } from '@/hooks/useStore';
import { Download } from 'lucide-react';

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

  // 5-34: Export PNG using canvas
  const handleExportPng = useCallback(() => {
    const root = document.getElementById('root');
    if (!root) return;
    import('html2canvas').then(({ default: html2canvas }) => {
      html2canvas(root, { backgroundColor: '#0a0e1a' }).then((canvas) => {
        const link = document.createElement('a');
        link.download = `skill-graph-${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
      });
    }).catch(() => {
      alert('Export failed — html2canvas not installed.');
    });
  }, []);

  return (
    <div className="h-screen flex flex-col bg-[hsl(var(--background))] text-[hsl(var(--foreground))]">
      {/* Header */}
      <Header />

      {/* Command Bar (interactive mode) */}
      {mode === 'interactive' && <CommandBar />}

      {/* Toolbar */}
      <div className="flex items-center justify-end px-4 py-1 gap-2 border-b border-[hsl(var(--border))]">
        <button
          onClick={handleExportPng}
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px] text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))] transition-colors"
          title="匯出 PNG"
        >
          <Download size={11} />
          Export PNG
        </button>
        <span className="text-[9px] text-[hsl(var(--muted-foreground))]">
          <kbd className="px-1 py-0.5 rounded bg-[hsl(var(--secondary))]">⌘K</kbd> search
        </span>
      </div>

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

      {/* Global Search Modal (Ctrl+K) */}
      <SearchModal />
    </div>
  );
}
