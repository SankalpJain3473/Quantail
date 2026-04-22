// src/components/Dashboard.tsx
import { useState } from 'react';
import { useTradingStore } from '../store/tradingStore';
import { sessionApi, dataApi } from '../lib/api';
import { KPICard } from './KPICard';
import { PnLChart } from './PnLChart';
import { PriceChart } from './PriceChart';
import { TradeTable } from './TradeTable';
import { AgentPanel } from './AgentPanel';
import { MarketPanel } from './MarketPanel';
import { VQCPanel } from './VQCPanel';
import { QuantailChat } from './QuantailChat';
import { SessionConfig } from '../types';

export function Dashboard() {
  const {
    stats, connected, sessionActive, trades, pnlHistory,
    auth, logout, setSessionActive, setSessionConfig, resetSession,
  } = useTradingStore();

  const [config, setConfig] = useState<SessionConfig>({
    symbol: 'SPY',
    mode: 'simulated',
    n_steps: 120,
    speed_ms: 1000,
    initial_capital: 100000,
  });
  const [loading, setLoading] = useState(false);
  const [alert, setAlert] = useState<{ type: 'risk' | 'info' | null; message: string }>({ type: null, message: '' });

  const startSession = async () => {
    setLoading(true);
    resetSession();
    try {
      await sessionApi.start(config);
      setSessionActive(true);
      setSessionConfig(config);
      setAlert({ type: 'info', message: `Session started — ${config.symbol} | ${config.mode} mode` });
      setTimeout(() => setAlert({ type: null, message: '' }), 3000);
    } catch (e: any) {
      setAlert({ type: 'risk', message: e.response?.data?.detail || 'Failed to start session' });
    } finally {
      setLoading(false);
    }
  };

  const stopSession = async () => {
    await sessionApi.stop();
    setSessionActive(false);
    setAlert({ type: 'info', message: 'Session stopped' });
    setTimeout(() => setAlert({ type: null, message: '' }), 3000);
  };

  const exportTrades = async () => {
    try {
      const { data } = await dataApi.exportTrades();
      const blob = new Blob([data.csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = data.filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setAlert({ type: 'risk', message: 'No trades to export yet' });
    }
  };

  const pnl = stats?.total_pnl ?? 0;
  const isProfit = pnl >= 0;

  return (
    <div className="min-h-screen bg-[#0f0f13] text-[#e0dfd8] text-[13px]">

      {/* Header */}
      <header className="bg-[#18181f] border-b border-[#2a2a35] px-4 py-2.5 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold">
            <span className="text-[#7f77dd]">Quant</span>
            <span className="text-[#5dcaa5]">ail</span>
          </h1>

          {/* Connection status */}
          <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-bold border ${
            connected
              ? 'bg-[#0a2e1e] text-[#5dcaa5] border-[#0f6e56]'
              : 'bg-[#1e1e28] text-[#888780] border-[#3a3a45]'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-[#5dcaa5] animate-pulse' : 'bg-[#888780]'}`} />
            {connected ? 'CONNECTED' : 'DISCONNECTED'}
          </div>

          {/* Mode badge */}
          <div className={`px-2 py-1 rounded text-xs font-bold border ${
            config.mode === 'live' ? 'bg-[#0a2e1e] text-[#5dcaa5] border-[#0f6e56]' :
            config.mode === 'paper' ? 'bg-[#2a2015] text-[#ef9f27] border-[#854f0b]' :
            'bg-[#1e1e28] text-[#888780] border-[#3a3a45]'
          }`}>
            {config.mode === 'live' ? '● LIVE DATA' : config.mode === 'paper' ? '● PAPER' : '● SIM'}
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <select
            value={config.symbol}
            onChange={(e) => setConfig({ ...config, symbol: e.target.value })}
            disabled={sessionActive}
            className="bg-[#0f0f13] border border-[#2a2a35] text-[#e0dfd8] px-2 py-1.5 rounded-md text-xs disabled:opacity-50"
          >
            {['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA'].map(s => <option key={s}>{s}</option>)}
          </select>

          <select
            value={config.mode}
            onChange={(e) => setConfig({ ...config, mode: e.target.value as SessionConfig['mode'] })}
            disabled={sessionActive}
            className="bg-[#0f0f13] border border-[#2a2a35] text-[#e0dfd8] px-2 py-1.5 rounded-md text-xs disabled:opacity-50"
          >
            <option value="simulated">Heston Simulation</option>
            <option value="paper">Paper Trade (real prices)</option>
            <option value="live">Live Data (Yahoo Finance)</option>
          </select>

          <select
            value={config.speed_ms}
            onChange={(e) => setConfig({ ...config, speed_ms: Number(e.target.value) })}
            disabled={sessionActive}
            className="bg-[#0f0f13] border border-[#2a2a35] text-[#e0dfd8] px-2 py-1.5 rounded-md text-xs disabled:opacity-50"
          >
            <option value={300}>Fast (300ms)</option>
            <option value={1000}>Normal (1s)</option>
            <option value={3000}>Slow (3s)</option>
          </select>

          {!sessionActive ? (
            <button
              onClick={startSession}
              disabled={loading || !connected}
              className="bg-[#0f6e56] hover:bg-[#0d5e49] text-[#9fe1cb] font-bold px-3 py-1.5 rounded-md text-xs transition-colors disabled:opacity-50"
            >
              {loading ? 'Starting...' : '▶ Start'}
            </button>
          ) : (
            <button
              onClick={stopSession}
              className="bg-[#3d1515] hover:bg-[#4d1f1f] text-[#f09595] font-bold px-3 py-1.5 rounded-md text-xs transition-colors"
            >
              ■ Stop
            </button>
          )}

          <button
            onClick={exportTrades}
            className="bg-[#1e1e28] hover:bg-[#2a2a35] text-[#7f77dd] font-bold px-3 py-1.5 rounded-md text-xs border border-[#534ab7] transition-colors"
          >
            ↓ Export CSV
          </button>

          <button
            onClick={logout}
            className="text-[#5f5e5a] hover:text-[#888780] text-xs px-2 py-1.5 transition-colors"
          >
            Sign out
          </button>
        </div>
      </header>

      {/* Alert banner */}
      {alert.type && (
        <div className={`px-4 py-2 text-xs font-semibold ${
          alert.type === 'risk'
            ? 'bg-[#3d1515] text-[#f09595] border-b border-[#a32d2d]'
            : 'bg-[#0a1e30] text-[#5dcaa5] border-b border-[#0f6e56]'
        }`}>
          {alert.message}
        </div>
      )}

      {/* CVaR alert */}
      {stats && stats.cvar_95 > 0.03 && (
        <div className="px-4 py-2 text-xs font-semibold bg-[#3d1515] text-[#f09595] border-b border-[#a32d2d]">
          ⊘ RISK ALERT — CVaR {stats.cvar_95.toFixed(4)} exceeds limit 0.03 · Risk agent veto active
        </div>
      )}

      <div className="p-3 space-y-3">

        {/* KPI Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <KPICard
            title="Total P&L"
            value={`${isProfit ? '+' : ''}$${Math.abs(pnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            sub={`${(stats?.return_pct ?? 0) >= 0 ? '+' : ''}${(stats?.return_pct ?? 0).toFixed(3)}% return`}
            color={isProfit ? 'green' : 'red'}
            bar={null}
          />
          <KPICard
            title="CVaR @ 95%"
            value={(stats?.cvar_95 ?? 0).toFixed(4)}
            sub={(stats?.cvar_95 ?? 0) < 0.03 ? 'Within budget — limit 0.03' : '⚠ LIMIT BREACHED'}
            color={(stats?.cvar_95 ?? 0) < 0.015 ? 'green' : (stats?.cvar_95 ?? 0) < 0.03 ? 'amber' : 'red'}
            bar={{ value: Math.min((stats?.cvar_95 ?? 0) / 0.06 * 100, 100), color: (stats?.cvar_95 ?? 0) < 0.03 ? 'green' : 'red' }}
          />
          <KPICard
            title="Sharpe / Sortino"
            value={(stats?.sharpe ?? 0).toFixed(3)}
            sub={`Sortino: ${(stats?.sortino ?? 0).toFixed(3)}`}
            color={(stats?.sharpe ?? 0) > 1 ? 'green' : (stats?.sharpe ?? 0) > 0 ? 'amber' : 'red'}
            bar={null}
          />
          <KPICard
            title="Hedge Ratio"
            value={(stats?.hedge_ratio ?? 0).toFixed(4)}
            sub={`Delta: ${(stats?.delta ?? 0).toFixed(4)} | ${stats?.n_trades ?? 0} trades`}
            color="purple"
            bar={{ value: ((stats?.hedge_ratio ?? 0) + 1) / 2 * 100, color: 'purple' }}
          />
        </div>

        {/* Charts + Panels row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
          <div className="lg:col-span-2 space-y-3">
            <PnLChart />
            <PriceChart />
          </div>
          <div className="space-y-3">
            <MarketPanel stats={stats} />
            <AgentPanel stats={stats} />
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[
            { label: 'Steps', value: stats?.step ?? 0 },
            { label: 'Trades', value: stats?.n_trades ?? 0 },
            { label: 'Tx Cost', value: `$${(stats?.total_cost ?? 0).toFixed(4)}` },
            { label: 'RMSE', value: (stats?.hedging_rmse ?? 0).toFixed(6) },
            { label: 'Veto rate', value: `${(stats?.veto_rate ?? 0).toFixed(1)}%` },
          ].map(({ label, value }) => (
            <div key={label} className="bg-[#18181f] border border-[#2a2a35] rounded-lg p-3">
              <div className="text-[10px] font-bold text-[#888780] uppercase tracking-wider mb-1">{label}</div>
              <div className="text-base font-bold text-[#e0dfd8]">{value}</div>
            </div>
          ))}
        </div>

        {/* Trade table */}
        <TradeTable trades={trades} />

      </div>

      {/* AI Chatbot */}
      <QuantailChat />

      {/* Footer */}
      <footer className="px-4 py-2 border-t border-[#1e1e28] flex justify-between text-[10px] text-[#5f5e5a] flex-wrap gap-1">
        <span>Quantail PoC · Sankalp Jain & Veronica Koval · Columbia University · YC S2026</span>
        <span>Step {stats?.step ?? 0} · {auth.username} · {new Date().toLocaleTimeString()}</span>
      </footer>
    </div>
  );
}
