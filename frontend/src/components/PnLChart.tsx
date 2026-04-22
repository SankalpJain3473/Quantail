// src/components/PnLChart.tsx
import { useTradingStore } from '../store/tradingStore';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export function PnLChart() {
  const { pnlHistory, stats } = useTradingStore();
  const pnl = stats?.total_pnl ?? 0;
  const color = pnl >= 0 ? '#5dcaa5' : '#f09595';
  const fillColor = pnl >= 0 ? 'rgba(93,202,165,0.08)' : 'rgba(240,149,149,0.08)';

  return (
    <div className="bg-[#18181f] border border-[#2a2a35] rounded-lg p-3">
      <div className="text-[10px] font-bold text-[#888780] uppercase tracking-wider mb-2">Portfolio Value</div>
      <ResponsiveContainer width="100%" height={140}>
        <AreaChart data={pnlHistory} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.15} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e28" />
          <XAxis dataKey="step" tick={false} axisLine={false} tickLine={false} />
          <YAxis
            tick={{ fill: '#888780', fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            width={52}
          />
          <Tooltip
            contentStyle={{ background: '#18181f', border: '1px solid #2a2a35', borderRadius: 8, fontSize: 11 }}
            labelStyle={{ color: '#888780' }}
            formatter={(value: number) => [`$${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`, 'Portfolio']}
          />
          <Area
            type="monotone"
            dataKey="total_value"
            stroke={color}
            strokeWidth={2}
            fill="url(#pnlGrad)"
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
