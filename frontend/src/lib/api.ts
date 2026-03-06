/**
 * REST API client — dynamic hostname for remote access.
 */

const API_BASE = `http://${window.location.hostname}:8000`;

export async function apiGet<T = unknown>(path: string): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
    return res.json();
}

export async function apiPost<T = unknown>(
    path: string,
    body: unknown,
): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
    return res.json();
}

export const WS_URL = `ws://${window.location.hostname}:8000/ws/live`;
