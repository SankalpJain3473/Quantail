// src/components/QuantailChat.tsx
// Floating AI chatbot — live session data + CSV file upload for analysis

import { useState, useRef, useEffect, useCallback } from "react";
import { useTradingStore } from "../store/tradingStore";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  loading?: boolean;
  filename?: string;
}

function buildSystemPrompt(stats: any, trades: any[], sessionActive: boolean): string {
  const liveContext = stats ? `
LIVE SESSION DATA (real-time):
- Step: ${stats.step}
- Total P&L: $${stats.total_pnl >= 0 ? "+" : ""}${stats.total_pnl?.toFixed(2)}
- Return: ${stats.return_pct?.toFixed(3)}%
- CVaR @ 95%: ${stats.cvar_95?.toFixed(4)} (limit: 0.03 — ${stats.cvar_95 < 0.03 ? "WITHIN BUDGET" : "⚠ BREACHED"})
- Sharpe ratio: ${stats.sharpe?.toFixed(3)}
- Sortino ratio: ${stats.sortino?.toFixed(3)}
- Hedge ratio: ${stats.hedge_ratio?.toFixed(4)}
- Delta: ${stats.delta?.toFixed(4)}
- Spot price: $${stats.spot_price?.toFixed(2)}
- IV: ${stats.iv?.toFixed(1)}%
- Trades executed: ${stats.n_trades}
- Hedging RMSE: ${stats.hedging_rmse?.toFixed(6)}
- Veto rate: ${stats.veto_rate?.toFixed(1)}%
- Data source: ${stats.data_source}
- Agent weights: ${JSON.stringify(stats.agent_weights)}
- Market regime: detected via RegimeAgent
- Session active: ${sessionActive}
` : "No active session. User has not started trading yet.";

  const tradeContext = trades.length > 0 ? `
RECENT TRADES (last ${Math.min(trades.length, 5)}):
${trades.slice(0, 5).map((t: any, i: number) =>
  `${i+1}. ${t.side?.toUpperCase()} ${t.qty} @ $${t.fill_price} | hedge ${t.hedge_before}→${t.hedge_after} | error: ${t.hedging_error} | cost: $${t.cost}`
).join("\n")}
` : "No trades yet.";

  return `You are the Quantail AI assistant — an expert quant analyst embedded directly in the Quantail trading dashboard.

WHAT YOU ARE:
Quantail is an institutional-grade algorithmic hedging platform built by Sankalp Jain and Veronica Koval at Columbia University (YC S2026 applicants). It uses:
- 5 specialized VQC (Variational Quantum Circuit) agents: HedgingAgent, RiskAgent, QuantumExplorerAgent, RegimeAgent, AlphaAgent
- Heston-Bates-Hamilton hybrid SDE (stochastic vol + price jumps + regime switching)
- Wasserstein W₂ barycenter coordination across agents
- CVaR @ 95% as the coherent risk measure (Artzner 1999 axioms)
- 13-qubit VQC with 8192 Fourier modes (O(2^13) expressivity vs O(poly(13)) for MLP)
- 4 market regimes: calm, stressed, crisis, recovery
- Invite-only access with JWT auth and bcrypt passwords

AGENT ROLES:
- HedgingAgent: minimizes hedging error ε² + transaction cost. "The execution trader"
- RiskAgent: monitors CVaR + drawdown + liquidity risk, hard veto when limits breached. "The risk officer"
- QuantumExplorerAgent: dedicated VQC action space explorer, diversity reward. "The quant researcher"
- RegimeAgent: classifies market regime, shifts coordinator weights dynamically. "The senior strategist"
- AlphaAgent: directional momentum signals. "The market strategist"

${liveContext}
${tradeContext}

HOW TO RESPOND:
- Be direct and specific. Use actual numbers from the live data above.
- You are talking to a quant researcher or trader — no hand-holding.
- If asked about P&L, give the exact number and explain what drove it.
- If asked about risk, reference CVaR, drawdown, veto rate specifically.
- If asked what to do, give a concrete recommendation.
- Keep responses concise — 2-4 sentences for simple questions, up to 8 sentences for complex ones.
- Use $, %, bps correctly. Never say "I don't have access to" — you have the live data above.
- If no session is active, help the user understand how to start one.
- If CSV data is provided in context, analyse it thoroughly: summarise columns, row count, key stats, anomalies, and answer questions about it.
- Never give financial advice for real money. This is a paper trading / research platform.`;
}

const SUGGESTIONS = [
  "Why is my P&L negative?",
  "Is my CVaR within budget?",
  "What is the RegimeAgent doing?",
  "Explain the last 3 trades",
  "Should I run more training?",
  "What does the veto rate mean?",
  "How is the VQC different from MLP?",
  "What caused the worst error?",
];

function renderContent(text: string) {
  const lines = text.split("\n");
  return lines.map((line, i) => {
    if (line.startsWith("**") && line.endsWith("**") && line.length > 4) {
      return <div key={i} style={{ fontWeight: 700, color: "#c8c4f8", marginTop: 4 }}>{line.slice(2,-2)}</div>;
    }
    if (line.startsWith("- ") || line.startsWith("• ")) {
      return (
        <div key={i} style={{ display: "flex", gap: 6, marginTop: 2 }}>
          <span style={{ color: "#4ecba4", flexShrink: 0, marginTop: 1 }}>▸</span>
          <span>{inlineBold(line.slice(2))}</span>
        </div>
      );
    }
    if (line.trim() === "") return <div key={i} style={{ height: 4 }} />;
    return <div key={i} style={{ marginTop: 2 }}>{inlineBold(line)}</div>;
  });
}

function inlineBold(text: string) {
  const parts = text.split(/\*\*(.*?)\*\*/g);
  return parts.map((p, i) =>
    i % 2 === 1
      ? <strong key={i} style={{ color: "#e0dfd8", fontWeight: 700 }}>{p}</strong>
      : <span key={i}>{p}</span>
  );
}


export function QuantailChat() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hey — I'm your Quantail trading assistant. I have live access to your session data, agent states, P&L, and trade history. You can also upload a CSV file for analysis.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [unread, setUnread] = useState(0);
  const [csvFile, setCsvFile] = useState<{ name: string; content: string } | null>(null);
  const [csvError, setCsvError] = useState("");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { stats, trades, sessionActive, auth } = useTradingStore();

  useEffect(() => {
    if (open) messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 100);
      setUnread(0);
    }
  }, [open]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setCsvError("");

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setCsvError("Only .csv files are supported.");
      e.target.value = "";
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setCsvError("File too large (max 5 MB).");
      e.target.value = "";
      return;
    }

    const reader = new FileReader();
    reader.onload = (ev) => {
      const content = ev.target?.result as string;
      setCsvFile({ name: file.name, content });
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const removeCsv = () => {
    setCsvFile(null);
    setCsvError("");
  };

  const send = useCallback(async (text?: string) => {
    const content = (text || input).trim();
    if (!content || loading) return;
    setInput("");

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
      filename: csvFile?.name,
    };
    const loadingMsg: Message = {
      id: Date.now().toString() + "_loading",
      role: "assistant",
      content: "",
      timestamp: new Date(),
      loading: true,
    };

    setMessages(prev => [...prev, userMsg, loadingMsg]);
    setLoading(true);

    // Attach CSV notice to user message if present
    const messageContent = csvFile
      ? `${content}\n\n[Attached CSV: ${csvFile.name}]`
      : content;

    const history = messages
      .filter(m => !m.loading)
      .map(m => ({ role: m.role, content: m.content }));
    history.push({ role: "user", content: messageContent });

    try {
      const token = auth.token;
      const response = await fetch(`${import.meta.env.VITE_API_URL || "http://localhost:8000"}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          messages: history,
          system: buildSystemPrompt(stats, trades, sessionActive),
          csv_content: csvFile?.content ?? "",
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      const reply = data.content || "Something went wrong. Please try again.";

      setMessages(prev =>
        prev.filter(m => !m.loading).concat({
          id: Date.now().toString() + "_reply",
          role: "assistant",
          content: reply,
          timestamp: new Date(),
        })
      );

      // Clear CSV after first use so it doesn't re-send every message
      setCsvFile(null);
      if (!open) setUnread(n => n + 1);

    } catch (e: any) {
      setMessages(prev =>
        prev.filter(m => !m.loading).concat({
          id: Date.now().toString() + "_err",
          role: "assistant",
          content: `Error: ${e.message || "Could not reach the server. Make sure the backend is running."}`,
          timestamp: new Date(),
        })
      );
    } finally {
      setLoading(false);
    }
  }, [input, loading, messages, stats, trades, sessionActive, open, csvFile, auth]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <>
      {/* Floating button */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          position: "fixed", bottom: 24, right: 24,
          width: 52, height: 52, borderRadius: "50%",
          background: open ? "#2a2838" : "linear-gradient(135deg, #534ab7, #3d3590)",
          border: `2px solid ${open ? "#3a3848" : "#7b73e8"}`,
          cursor: "pointer", display: "flex", alignItems: "center",
          justifyContent: "center",
          boxShadow: open ? "none" : "0 4px 20px rgba(83,74,183,0.4)",
          transition: "all 0.2s ease", zIndex: 1000, userSelect: "none",
        }}
        title="Quantail AI assistant"
      >
        {open ? (
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d="M2 2L16 16M16 2L2 16" stroke="#888798" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        ) : (
          <span style={{ fontSize: 18, fontWeight: 900, color: "white", fontFamily: "serif" }}>Q</span>
        )}
        {unread > 0 && !open && (
          <div style={{
            position: "absolute", top: -4, right: -4,
            width: 18, height: 18, borderRadius: "50%",
            background: "#e05555", fontSize: 10, fontWeight: 700,
            color: "white", display: "flex", alignItems: "center",
            justifyContent: "center", border: "2px solid #0b0b0f",
          }}>{unread}</div>
        )}
        {sessionActive && !open && (
          <div style={{
            position: "absolute", inset: -4, borderRadius: "50%",
            border: "2px solid rgba(78,203,164,0.4)",
            animation: "chatPulse 2s ease-in-out infinite",
          }} />
        )}
      </div>

      {/* Chat window */}
      {open && (
        <div style={{
          position: "fixed", bottom: 88, right: 24,
          width: 360, height: 540,
          background: "#13131a", border: "1px solid #1e1e2a",
          borderRadius: 14, display: "flex", flexDirection: "column",
          boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
          zIndex: 999, overflow: "hidden",
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          animation: "chatSlideUp 0.2s ease",
        }}>

          {/* Header */}
          <div style={{
            padding: "12px 14px", borderBottom: "1px solid #1e1e2a",
            display: "flex", alignItems: "center", gap: 10,
            background: "#0f0f15", flexShrink: 0,
          }}>
            <div style={{
              width: 28, height: 28, borderRadius: 7,
              background: "linear-gradient(135deg,#534ab7,#3d3590)",
              display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
            }}>
              <span style={{ fontSize: 12, fontWeight: 900, color: "white" }}>Q</span>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#c8c4f8" }}>Quantail AI</div>
              <div style={{ fontSize: 10, color: sessionActive ? "#4ecba4" : "#888798", display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{
                  width: 5, height: 5, borderRadius: "50%",
                  background: sessionActive ? "#4ecba4" : "#888798",
                  display: "inline-block",
                  ...(sessionActive ? { animation: "chatPulse 1.5s infinite" } : {}),
                }} />
                {sessionActive ? `Live · Step ${stats?.step ?? 0}` : "No active session"}
              </div>
            </div>
            {stats && (
              <div style={{
                fontSize: 11, fontWeight: 700,
                color: (stats.total_pnl ?? 0) >= 0 ? "#4ecba4" : "#e05555",
                background: (stats.total_pnl ?? 0) >= 0 ? "#0a2e1e" : "#2e0a0a",
                padding: "3px 8px", borderRadius: 5,
                border: `1px solid ${(stats.total_pnl ?? 0) >= 0 ? "#0f6e56" : "#6e2020"}`,
              }}>
                {(stats.total_pnl ?? 0) >= 0 ? "+" : ""}${stats.total_pnl?.toFixed(0)}
              </div>
            )}
          </div>

          {/* Messages */}
          <div style={{
            flex: 1, overflowY: "auto", padding: "12px 12px 4px",
            display: "flex", flexDirection: "column", gap: 10,
          }}>
            {messages.map(msg => (
              <div key={msg.id} style={{
                display: "flex",
                flexDirection: msg.role === "user" ? "row-reverse" : "row",
                gap: 8, alignItems: "flex-end",
              }}>
                {msg.role === "assistant" && (
                  <div style={{
                    width: 22, height: 22, borderRadius: "50%",
                    background: "linear-gradient(135deg,#534ab7,#3d3590)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    flexShrink: 0, marginBottom: 1,
                  }}>
                    <span style={{ fontSize: 9, fontWeight: 900, color: "white" }}>Q</span>
                  </div>
                )}
                <div style={{ maxWidth: "78%" }}>
                  {msg.filename && (
                    <div style={{
                      fontSize: 9.5, color: "#4ecba4", marginBottom: 3,
                      display: "flex", alignItems: "center", gap: 4,
                      justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                    }}>
                      <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                        <rect x="1" y="1" width="10" height="10" rx="2" stroke="#4ecba4" strokeWidth="1.2"/>
                        <path d="M3 4h6M3 6h4" stroke="#4ecba4" strokeWidth="1.2" strokeLinecap="round"/>
                      </svg>
                      {msg.filename}
                    </div>
                  )}
                  <div style={{
                    padding: "8px 11px",
                    borderRadius: msg.role === "user" ? "12px 12px 3px 12px" : "12px 12px 12px 3px",
                    background: msg.role === "user"
                      ? "linear-gradient(135deg,#534ab7,#3d3590)"
                      : "#1e1e2e",
                    border: msg.role === "assistant" ? "1px solid #2a2a3a" : "none",
                    fontSize: 11.5, lineHeight: 1.55,
                    color: msg.role === "user" ? "#e8e6ff" : "#b8b6d0",
                  }}>
                    {msg.loading ? (
                      <div style={{ display: "flex", gap: 4, alignItems: "center", padding: "2px 4px" }}>
                        {[0, 1, 2].map(i => (
                          <div key={i} style={{
                            width: 5, height: 5, borderRadius: "50%",
                            background: "#7b73e8",
                            animation: `chatDot 1.2s ease-in-out ${i * 0.2}s infinite`,
                          }} />
                        ))}
                      </div>
                    ) : (
                      <div>{renderContent(msg.content)}</div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Suggestions */}
          {messages.length === 1 && (
            <div style={{ padding: "4px 12px 8px", display: "flex", flexWrap: "wrap", gap: 5, flexShrink: 0 }}>
              {SUGGESTIONS.slice(0, 4).map(q => (
                <button key={q} onClick={() => send(q)} style={{
                  background: "#1a1a2a", border: "1px solid #2a2a3a", borderRadius: 6,
                  padding: "4px 9px", fontSize: 10, color: "#888798",
                  cursor: "pointer", fontFamily: "inherit", transition: "all 0.15s",
                }}
                  onMouseEnter={e => { (e.target as HTMLElement).style.borderColor = "#534ab7"; (e.target as HTMLElement).style.color = "#c8c4f8"; }}
                  onMouseLeave={e => { (e.target as HTMLElement).style.borderColor = "#2a2a3a"; (e.target as HTMLElement).style.color = "#888798"; }}
                >{q}</button>
              ))}
            </div>
          )}

          {/* CSV pill + error */}
          {(csvFile || csvError) && (
            <div style={{ padding: "0 12px 6px", flexShrink: 0 }}>
              {csvFile && (
                <div style={{
                  display: "inline-flex", alignItems: "center", gap: 6,
                  background: "#0a2e1e", border: "1px solid #0f6e56",
                  borderRadius: 6, padding: "4px 8px", fontSize: 10, color: "#4ecba4",
                }}>
                  <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                    <rect x="1" y="1" width="10" height="10" rx="2" stroke="#4ecba4" strokeWidth="1.2"/>
                    <path d="M3 4h6M3 6h4" stroke="#4ecba4" strokeWidth="1.2" strokeLinecap="round"/>
                  </svg>
                  {csvFile.name}
                  <span style={{ color: "#888798", fontSize: 9 }}>
                    ({(csvFile.content.length / 1024).toFixed(1)}KB)
                  </span>
                  <span
                    onClick={removeCsv}
                    style={{ cursor: "pointer", color: "#888798", marginLeft: 2, lineHeight: 1 }}
                    title="Remove"
                  >✕</span>
                </div>
              )}
              {csvError && (
                <div style={{ fontSize: 10, color: "#e05555", marginTop: 2 }}>{csvError}</div>
              )}
            </div>
          )}

          {/* Input row */}
          <div style={{ padding: "10px 12px", borderTop: "1px solid #1e1e2a", background: "#0f0f15", flexShrink: 0 }}>
            <div style={{
              display: "flex", gap: 6, alignItems: "flex-end",
              background: "#1a1a2a", border: "1px solid #2a2a3a",
              borderRadius: 10, padding: "6px 8px 6px 10px",
            }}>
              {/* CSV upload button */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                title="Upload CSV for analysis"
                style={{
                  background: csvFile ? "#0a2e1e" : "transparent",
                  border: `1px solid ${csvFile ? "#0f6e56" : "#2a2a3a"}`,
                  borderRadius: 6, width: 26, height: 26, flexShrink: 0,
                  cursor: "pointer", display: "flex", alignItems: "center",
                  justifyContent: "center", transition: "all 0.15s",
                }}
                onMouseEnter={e => { if (!csvFile) (e.currentTarget).style.borderColor = "#534ab7"; }}
                onMouseLeave={e => { if (!csvFile) (e.currentTarget).style.borderColor = "#2a2a3a"; }}
              >
                <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
                  <rect x="1" y="1" width="12" height="12" rx="2.5" stroke={csvFile ? "#4ecba4" : "#555"} strokeWidth="1.3"/>
                  <path d="M4 5h6M4 7h4M4 9h5" stroke={csvFile ? "#4ecba4" : "#555"} strokeWidth="1.3" strokeLinecap="round"/>
                </svg>
              </button>

              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder={csvFile ? `Ask about ${csvFile.name}...` : "Ask about P&L, risk, agents..."}
                rows={1}
                disabled={loading}
                style={{
                  flex: 1, background: "transparent", border: "none", outline: "none",
                  color: "#e0dfd8", fontSize: 11.5, fontFamily: "inherit",
                  resize: "none", lineHeight: 1.5, maxHeight: 80, paddingTop: 2,
                }}
                onInput={e => {
                  const t = e.target as HTMLTextAreaElement;
                  t.style.height = "auto";
                  t.style.height = Math.min(t.scrollHeight, 80) + "px";
                }}
              />
              <button
                onClick={() => send()}
                disabled={!input.trim() || loading}
                style={{
                  width: 28, height: 28, borderRadius: 7, flexShrink: 0,
                  background: input.trim() && !loading
                    ? "linear-gradient(135deg,#534ab7,#3d3590)"
                    : "#2a2a38",
                  border: "none",
                  cursor: input.trim() && !loading ? "pointer" : "default",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  transition: "background 0.15s",
                }}
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M1 11L11 1M11 1H4M11 1V8"
                    stroke={input.trim() && !loading ? "white" : "#555"}
                    strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>
            <div style={{ fontSize: 9, color: "#44445a", textAlign: "center", marginTop: 5 }}>
              Enter to send · Shift+Enter for newline · CSV icon to upload data
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes chatPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.15); }
        }
        @keyframes chatSlideUp {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes chatDot {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30% { transform: translateY(-4px); opacity: 1; }
        }
      `}</style>
    </>
  );
}
