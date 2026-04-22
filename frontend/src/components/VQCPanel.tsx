// src/components/VQCPanel.tsx
// Shows the quantum circuit internals — what each VQC agent is actually doing

import { useState, useEffect } from 'react';
import { api } from '../lib/api';
import { useTradingStore } from '../store/tradingStore';

interface AgentInfo {
  action: number;
  hedge_adj: number;
  top_prob: number;
  action_probs: number[];
  measurements: number[];
  cvar_dist: number;
  dist_mean: number;
  dist_std: number;
}

interface VQCInfo {
  ready: boolean;
  trained: boolean;
  step_count: number;
  agents: Record<string, AgentInfo>;
  expressivity: {
    n_qubits: number;
    n_layers: number;
    vqc_fourier_modes: number;
    mlp_fourier_modes: number;
    expressivity_ratio: number;
    n_params: number;
  };
}

const AGENT_COLORS: Record<string, string> = {
  HedgingAgent:   '#534ab7',
  RiskAgent:      '#0f6e56',
  PortfolioAgent: '#854f0b',
  AlphaAgent:     '#993c1d',
};

const ACTION_LABELS: Record<number, string> = {
  0: '−5%', 1: '−4%', 2: '−3%', 3: '−2%', 4: '−1%',
  5: '0%', 6: '+1%', 7: '+2%', 8: '+3%', 9: '+4%', 10: '+5%',
};

function MiniBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="bg-[#0f0f13] rounded-full h-1 overflow-hidden flex-1">
      <div
        className="h-full rounded-full transition-all duration-300"
        style={{ width: `${pct}%`, background: color }}
      />
    </div>
  );
}

function ActionProbBars({ probs, action }: { probs: number[]; action: number }) {
  return (
    <div className="space-y-0.5 mt-1">
      {probs.map((p, i) => (
        <div key={i} className="flex items-center gap-1">
          <span className={`text-[9px] w-6 text-right ${i === action ? 'text-[#ef9f27] font-bold' : 'text-[#5f5e5a]'}`}>
            {ACTION_LABELS[i]}
          </span>
          <div className="flex-1 bg-[#0f0f13] rounded-full h-1 overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${p * 100}%`,
                background: i === action ? '#ef9f27' : '#2a2a35',
              }}
            />
          </div>
          {i === action && (
            <span className="text-[9px] text-[#ef9f27] w-8">{(p * 100).toFixed(0)}%</span>
          )}
        </div>
      ))}
    </div>
  );
}

export function VQCPanel() {
  const { stats, sessionActive } = useTradingStore();
  const [vqcInfo, setVqcInfo] = useState<VQCInfo | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sessionActive && !stats) return;
    const fetchVQC = async () => {
      try {
        const { data } = await api.get('/api/agents/info');
        setVqcInfo(data);
      } catch {
        // Silently fail — agents may still be training
      }
    };
    fetchVQC();
    const interval = setInterval(fetchVQC, 2000);
    return () => clearInterval(interval);
  }, [sessionActive, stats?.step]);

  if (!vqcInfo) {
    return (
      <div className="bg-[#18181f] border border-[#2a2a35] rounded-lg p-3">
        <div className="text-[10px] font-bold text-[#888780] uppercase tracking-wider mb-2">
          VQC Policy Network
        </div>
        <div className="text-[11px] text-[#5f5e5a] text-center py-4">
          {sessionActive ? 'Loading VQC data...' : 'Start a session to see VQC decisions'}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#18181f] border border-[#2a2a35] rounded-lg p-3 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="text-[10px] font-bold text-[#888780] uppercase tracking-wider">
          VQC Policy Network — Live Decisions
        </div>
        <div className={`text-[10px] font-bold px-2 py-0.5 rounded ${
          vqcInfo.ready
            ? 'bg-[#0a2e1e] text-[#5dcaa5] border border-[#0f6e56]'
            : 'bg-[#2a2015] text-[#ef9f27] border border-[#854f0b]'
        }`}>
          {vqcInfo.ready ? '● TRAINED' : '◐ TRAINING...'}
        </div>
      </div>

      {/* Expressivity badge */}
      {vqcInfo.expressivity?.n_qubits && (
        <div className="grid grid-cols-4 gap-1.5">
          {[
            { label: 'Qubits', value: vqcInfo.expressivity.n_qubits },
            { label: 'Layers', value: vqcInfo.expressivity.n_layers },
            { label: 'Fourier modes', value: `O(${vqcInfo.expressivity.vqc_fourier_modes})` },
            { label: 'vs MLP', value: `${vqcInfo.expressivity.expressivity_ratio?.toFixed(0)}×` },
          ].map(({ label, value }) => (
            <div key={label} className="bg-[#0f0f13] rounded p-1.5 text-center">
              <div className="text-[11px] font-bold text-[#7f77dd]">{value}</div>
              <div className="text-[9px] text-[#5f5e5a]">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Per-agent VQC output */}
      {vqcInfo.agents && Object.entries(vqcInfo.agents)
        .filter(([k]) => k !== 'coordinator_weights' && k !== 'veto_rate' && k !== 'expressivity')
        .map(([name, info]: [string, AgentInfo]) => (
          <div key={name} className="border border-[#1e1e28] rounded-lg p-2.5">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ background: AGENT_COLORS[name] || '#534ab7' }}
                />
                <span className="text-[11px] font-semibold text-[#c2c0b6]">
                  {name.replace('Agent', '')}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-[#888780]">action:</span>
                <span className={`text-[11px] font-bold px-1.5 py-0.5 rounded ${
                  info.hedge_adj > 0 ? 'bg-[#0a2e1e] text-[#5dcaa5]' :
                  info.hedge_adj < 0 ? 'bg-[#3d1515] text-[#f09595]' :
                  'bg-[#1e1e28] text-[#888780]'
                }`}>
                  {info.hedge_adj >= 0 ? '+' : ''}{(info.hedge_adj * 100).toFixed(0)}% hedge
                </span>
              </div>
            </div>

            {/* Action probability distribution */}
            <div className="mb-2">
              <div className="text-[9px] text-[#5f5e5a] mb-1">Action probabilities (VQC softmax output)</div>
              <ActionProbBars probs={info.action_probs} action={info.action} />
            </div>

            {/* Qubit measurements */}
            <div className="mb-1.5">
              <div className="text-[9px] text-[#5f5e5a] mb-1">Pauli-Z measurements (8 qubits)</div>
              <div className="flex gap-0.5">
                {info.measurements.map((m, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-sm"
                    style={{
                      height: '20px',
                      background: m > 0
                        ? `rgba(83,74,183,${Math.abs(m)})`
                        : `rgba(240,149,149,${Math.abs(m)})`,
                      border: '1px solid #2a2a35',
                    }}
                    title={`Qubit ${i}: ${m.toFixed(3)}`}
                  />
                ))}
              </div>
              <div className="flex justify-between text-[9px] text-[#5f5e5a] mt-0.5">
                <span>Q1</span><span>Q4</span><span>Q8</span>
              </div>
            </div>

            {/* Return distribution stats */}
            <div className="flex gap-2 text-[10px]">
              <span className="text-[#888780]">Dist mean:</span>
              <span className={info.dist_mean >= 0 ? 'text-[#5dcaa5]' : 'text-[#f09595]'}>
                {info.dist_mean >= 0 ? '+' : ''}{info.dist_mean.toFixed(4)}
              </span>
              <span className="text-[#888780] ml-2">CVaR@5%:</span>
              <span className={info.cvar_dist < 0.02 ? 'text-[#5dcaa5]' : 'text-[#ef9f27]'}>
                {info.cvar_dist.toFixed(4)}
              </span>
            </div>
          </div>
        ))}

      {/* Online learning indicator */}
      <div className="flex items-center justify-between pt-1 border-t border-[#1e1e28]">
        <div className="text-[10px] text-[#888780]">
          Online learning steps: {vqcInfo.step_count}
        </div>
        <div className="text-[10px] text-[#5dcaa5]">
          {vqcInfo.trained ? '✓ Agents learning from live trades' : 'Warming up...'}
        </div>
      </div>
    </div>
  );
}
