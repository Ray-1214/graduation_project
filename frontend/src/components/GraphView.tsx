import { useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { useStore } from '@/hooks/useStore';
import type { GraphNode, GraphEdge } from '@/types';

const TIER_COLORS: Record<string, string> = {
    active: '#22c55e',
    cold: '#eab308',
    archive: '#6b7280',
};

const EDGE_COLORS: Record<string, string> = {
    co_occurrence: '#3b82f6',
    dependency: '#ef4444',
    abstraction: '#a855f7',
};

export function GraphView() {
    const graph = useStore((s) => s.graph);
    const svgRef = useRef<SVGSVGElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const draw = useCallback(() => {
        if (!svgRef.current || !containerRef.current || !graph) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const { width, height } = containerRef.current.getBoundingClientRect();
        svg.attr('width', width).attr('height', height);

        // Prepare data (deep copy to avoid D3 mutation issues)
        const nodes: GraphNode[] = graph.nodes.map((n) => ({ ...n }));
        const edges: GraphEdge[] = graph.edges.map((e) => ({
            source: e.src || (typeof e.source === 'string' ? e.source : (e.source as GraphNode).id),
            target: e.dst || (typeof e.target === 'string' ? e.target : (e.target as GraphNode).id),
            weight: e.weight || 1,
            type: e.type || 'co_occurrence',
        }));

        if (nodes.length === 0) {
            svg
                .append('text')
                .attr('x', width / 2)
                .attr('y', height / 2)
                .attr('text-anchor', 'middle')
                .attr('fill', 'hsl(215, 16%, 47%)')
                .attr('font-size', '13px')
                .text('No skills yet — run a task to populate the graph');
            return;
        }

        // Zoom
        const g = svg.append('g');
        svg.call(
            d3.zoom<SVGSVGElement, unknown>()
                .scaleExtent([0.2, 4])
                .on('zoom', (event) => g.attr('transform', event.transform)),
        );

        // Simulation
        const simulation = d3
            .forceSimulation(nodes as d3.SimulationNodeDatum[])
            .force(
                'link',
                d3
                    .forceLink(edges as d3.SimulationLinkDatum<d3.SimulationNodeDatum>[])
                    .id((d: unknown) => (d as GraphNode).id)
                    .distance(80),
            )
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(25));

        // Edges
        const link = g
            .append('g')
            .selectAll('line')
            .data(edges)
            .join('line')
            .attr('stroke', (d) => EDGE_COLORS[d.type || 'co_occurrence'] || '#3b82f6')
            .attr('stroke-width', (d) => Math.max(1, (d.weight || 1) * 2))
            .attr('stroke-opacity', 0.5);

        // Nodes
        const node = g
            .append('g')
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('r', (d) => Math.max(6, (d.utility || 0.5) * 16))
            .attr('fill', (d) => TIER_COLORS[d.tier || 'active'] || '#22c55e')
            .attr('stroke', '#000')
            .attr('stroke-width', 0.5)
            .attr('cursor', 'grab')
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            .call(
                d3
                    .drag<SVGCircleElement, GraphNode>()
                    .on('start', (event, d) => {
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    })
                    .on('drag', (event, d) => {
                        d.fx = event.x;
                        d.fy = event.y;
                    })
                    .on('end', (event, d) => {
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }) as any,
            );

        // Labels
        const labels = g
            .append('g')
            .selectAll('text')
            .data(nodes)
            .join('text')
            .text((d) => d.name || d.id)
            .attr('font-size', '10px')
            .attr('fill', 'hsl(210, 20%, 82%)')
            .attr('text-anchor', 'middle')
            .attr('dy', -12)
            .attr('pointer-events', 'none');

        // Tooltip
        node
            .append('title')
            .text(
                (d) =>
                    [
                        d.name || d.id,
                        `utility: ${d.utility?.toFixed(2) ?? '—'}`,
                        `tier: ${d.tier || '—'}`,
                        `frequency: ${d.frequency ?? '—'}`,
                        `policy: ${(d as any).policy ?? '—'}`,
                        `reinforcement: ${(d as any).reinforcement ?? '—'}`,
                        `cost: ${(d as any).cost ?? '—'}`,
                        `version: ${(d as any).version ?? '—'}`,
                    ].join('\n'),
            );

        // Tick
        simulation.on('tick', () => {
            link
                .attr('x1', (d) => (d.source as unknown as GraphNode).x ?? 0)
                .attr('y1', (d) => (d.source as unknown as GraphNode).y ?? 0)
                .attr('x2', (d) => (d.target as unknown as GraphNode).x ?? 0)
                .attr('y2', (d) => (d.target as unknown as GraphNode).y ?? 0);

            node.attr('cx', (d) => d.x ?? 0).attr('cy', (d) => d.y ?? 0);

            labels.attr('x', (d) => d.x ?? 0).attr('y', (d) => d.y ?? 0);
        });
    }, [graph]);

    useEffect(() => {
        draw();
        const handleResize = () => draw();
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, [draw]);

    return (
        <div ref={containerRef} className="w-full h-full graph-container relative">
            {/* Header */}
            <div className="absolute top-2 left-3 z-10 flex items-center gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]">
                    Skill Graph
                </span>
                {graph && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-[hsl(var(--secondary))] text-[hsl(var(--muted-foreground))]">
                        {graph.sigma_size} skills · H={graph.entropy.toFixed(2)}
                    </span>
                )}
            </div>

            {/* Legend */}
            <div className="absolute bottom-2 left-3 z-10 flex items-center gap-3 text-[10px] text-[hsl(var(--muted-foreground))]">
                {Object.entries(TIER_COLORS).map(([tier, color]) => (
                    <span key={tier} className="flex items-center gap-1">
                        <span
                            className="w-2.5 h-2.5 rounded-full"
                            style={{ background: color }}
                        />
                        {tier}
                    </span>
                ))}
            </div>

            <svg ref={svgRef} />
        </div>
    );
}
