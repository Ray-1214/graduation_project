import { useState, useRef, useEffect } from 'react';
import { Send, Brain, Zap, RefreshCw } from 'lucide-react';
import { useStore } from '@/hooks/useStore';
import { cn } from '@/lib/utils';

const MODES = [
    { key: 'auto', label: '自動', icon: RefreshCw },
    { key: 'learn', label: '學習', icon: Brain },
    { key: 'execute', label: '執行', icon: Zap },
] as const;

export function CommandBar() {
    const [input, setInput] = useState('');
    const [taskMode, setTaskMode] = useState<string>('auto');
    const runTask = useStore((s) => s.runTask);
    const isRunning = useStore((s) => s.isRunning);
    const commandHistory = useStore((s) => s.commandHistory);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const [historyIdx, setHistoryIdx] = useState(-1);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
        }
    }, [input]);

    const handleSubmit = () => {
        const text = input.trim();
        if (!text || isRunning) return;
        setInput('');
        setHistoryIdx(-1);
        runTask(text, taskMode);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
        // History navigation
        if (e.key === 'ArrowUp' && !input) {
            e.preventDefault();
            const idx = Math.min(historyIdx + 1, commandHistory.length - 1);
            if (commandHistory[commandHistory.length - 1 - idx]) {
                setInput(commandHistory[commandHistory.length - 1 - idx].command);
                setHistoryIdx(idx);
            }
        }
        if (e.key === 'ArrowDown' && historyIdx >= 0) {
            e.preventDefault();
            const idx = historyIdx - 1;
            if (idx < 0) {
                setInput('');
                setHistoryIdx(-1);
            } else if (commandHistory[commandHistory.length - 1 - idx]) {
                setInput(commandHistory[commandHistory.length - 1 - idx].command);
                setHistoryIdx(idx);
            }
        }
    };

    return (
        <div className="px-5 py-3 glass border-b border-[hsl(var(--border))]">
            <div className="flex items-end gap-3">
                {/* Mode selector */}
                <div className="flex flex-col gap-1">
                    <span className="text-[10px] text-[hsl(var(--muted-foreground))] uppercase tracking-wider">
                        Mode
                    </span>
                    <div className="flex items-center bg-[hsl(var(--secondary))] rounded-md p-0.5">
                        {MODES.map(({ key, label, icon: Icon }) => (
                            <button
                                key={key}
                                onClick={() => setTaskMode(key)}
                                className={cn(
                                    'flex items-center gap-1 px-2 py-1 rounded text-xs transition-all',
                                    taskMode === key
                                        ? 'bg-[hsl(var(--primary))] text-white'
                                        : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]',
                                )}
                            >
                                <Icon size={11} />
                                {label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Input */}
                <div className="flex-1 relative">
                    <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="輸入指令… 例：「學習 Python 裝飾器」、「計算 15 * 23」"
                        rows={1}
                        disabled={isRunning}
                        className={cn(
                            'w-full resize-none rounded-lg border border-[hsl(var(--border))]',
                            'bg-[hsl(var(--secondary))] px-4 py-2.5 pr-12',
                            'text-sm text-[hsl(var(--foreground))]',
                            'placeholder:text-[hsl(var(--muted-foreground))]',
                            'focus:outline-none focus:ring-2 focus:ring-[hsl(var(--ring))] focus:border-transparent',
                            'transition-all',
                            isRunning && 'opacity-50 cursor-not-allowed',
                        )}
                    />
                    {isRunning && (
                        <span className="absolute left-4 -top-5 text-xs text-blue-400 animate-pulse-dot">
                            ⏳ 正在處理…
                        </span>
                    )}
                </div>

                {/* Send button */}
                <button
                    onClick={handleSubmit}
                    disabled={isRunning || !input.trim()}
                    className={cn(
                        'flex items-center justify-center w-10 h-10 rounded-lg transition-all',
                        'bg-[hsl(var(--primary))] text-white',
                        'hover:brightness-110 active:scale-95',
                        (isRunning || !input.trim()) && 'opacity-40 cursor-not-allowed',
                    )}
                >
                    <Send size={16} />
                </button>
            </div>

            {/* 5-39: Command history */}
            {commandHistory.length > 0 && (
                <div className="flex items-center gap-2 mt-2 overflow-x-auto pb-1">
                    {commandHistory.slice(-5).reverse().map((cmd, i) => (
                        <button
                            key={i}
                            onClick={() => setInput(cmd.command)}
                            className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-[hsl(var(--secondary))] text-[10px] text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))] transition-colors shrink-0 max-w-[200px]"
                        >
                            <span className="px-1 py-0 rounded bg-[hsl(var(--accent))] text-[9px]">
                                {cmd.mode}
                            </span>
                            <span className="truncate">{cmd.command}</span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
