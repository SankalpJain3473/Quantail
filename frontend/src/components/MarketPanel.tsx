// src/components/MarketPanel.tsx
import { SessionStats } from '../types';
interface Props { stats: SessionStats | null }

export function MarketPanel({ stats }: Props) {
  const rows = [
    { label: 'Spot price', value: stats ? `$${stats.spot_price.toFixed(2)}` : '—', color: 'text-[#e0dfd8]' },
    { label: 'Bid / Ask', value: stats ? `$${stats.bid.toFixed(2)} / $${stats.ask.toFixed(2)}` : '—', color: 'text-[#c2c0b6]' },
    { label: 'IV (annual)', value: stats ? `${stats.iv.toFixed(1)}%` : '—', color: 'text-[#ef9f27]' },
    { label: 'Delta (Δ)', value: stats ? stats.delta.toFixed(4) : '—', color: 'text-[#7f77dd]' },
    { label: 'Gamma (Γ)', value: stats ? stats.gamma.toFixed(6) : '—', color: 'text-[#7f77dd]' },
    { label: 'Data source', value: stats?.data_source ?? '—', color: 'text-[#5dcaa5]' },
  ];

  return (
    <div className="bg-[#18181f] border border-[#2a2a35] rounded-lg p-3">
      <div className="text-[10px] font-bold text-[#888780] uppercase tracking-wider mb-3">Live Market</div>
      <div className="space-y-1.5">
        {rows.map(({ label, value, color }) => (
          <div key={label} className="flex justify-between items-center py-1 border-b border-[#1a1a22] last:border-0">
            <span className="text-[11px] text-[#888780]">{label}</span>
            <span className={`text-[11px] font-semibold ${color}`}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
