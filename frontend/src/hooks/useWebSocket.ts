/**
 * WebSocket hook — auto-reconnect + event dispatch to store.
 */

import { useEffect, useRef } from 'react';
import { WS_URL } from '@/lib/api';
import { useStore } from '@/hooks/useStore';

export function useWebSocket() {
    const addWSEvent = useStore((s) => s.addWSEvent);
    const wsRef = useRef<WebSocket | null>(null);
    const retryRef = useRef(0);

    useEffect(() => {
        function connect() {
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                retryRef.current = 0;
                console.log('[WS] connected');
            };

            ws.onmessage = (evt) => {
                try {
                    const data = JSON.parse(evt.data);
                    if (data !== 'pong') {
                        addWSEvent(data);
                    }
                } catch { /* ignore non-JSON */ }
            };

            ws.onclose = () => {
                const delay = Math.min(1000 * 2 ** retryRef.current, 10000);
                retryRef.current += 1;
                console.log(`[WS] closed — reconnect in ${delay}ms`);
                setTimeout(connect, delay);
            };

            ws.onerror = () => ws.close();
        }

        connect();

        return () => {
            wsRef.current?.close();
        };
    }, [addWSEvent]);
}
