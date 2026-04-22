// src/components/AgentPanel.tsx
import { SessionStats } from '../types';

interface Props { stats: SessionStats | null }

const AGENT_COLORS: Record<string, string> = {
  HedgingAgent: '#534ab7',
  RiskAgent: '#0f6e56',
  PortfolioAgent: '#854f0b',
  AlphaAgent: '#993c1d',
};

export function AgentPanel({ stats }: Props) {
  const weights = stats?.agent_weights ?? {
    HedgingAgent: 0.40, RiskAgent: 0.30, PortfolioAgent: 0.20, AlphaAgent: 0.10,
  };

  return (
    <div className="bg-[#18181f] border border-[#2a2a35] rounded-lg p-3">
      <div className="text-[10px] font-bold text-[#888780] uppercase tracking-wider mb-3">
        Agent Weights — Wasserstein W₂
      </div>
      <div className="space-y-2.5">
        {Object.entries(weights).map(([name, weight]) => (
          <div key={name} className="flex items-center gap-2">
            <span className="text-[11px] text-[#c2c0b6] w-28 truncate">{name.replace('Agent', '')}</span>
            <div className="flex-1 bg-[#0f0f13] rounded-full h-1.5 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${weight * 100}%`, background: AGENT_COLORS[name] || '#534ab7' }}
              />
            </div>
            <span className="text-[11px] font-bold text-[#7f77dd] w-8 text-right">
              {Math.round(weight * 100)}%
            </span>
          </div>
        ))}
      </div>
      <div className="mt-3 pt-2 border-t border-[#1e1e28] text-[10px] text-[#888780]">
        Veto rate: {(stats?.veto_rate ?? 0).toFixed(1)}% · RMSE: {(stats?.hedging_rmse ?? 0).toFixed(4)}
      </div>
    </div>
  );
}
