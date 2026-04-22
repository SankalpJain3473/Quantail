// src/components/PriceChart.tsx
import { useTradingStore } from '../store/tradingStore';
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export function PriceChart() {
  const { priceHistory } = useTradingStore();

  return (
    <div className="bg-[#18181f] border border-[#2a2a35] rounded-lg p-3">
      <div className="text-[10px] font-bold text-[#888780] uppercase tracking-wider mb-2">Price & Hedge Ratio</div>
      <ResponsiveContainer width="100%" height={110}>
        <ComposedChart data={priceHistory} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e28" />
          <XAxis dataKey="step" tick={false} axisLine={false} tickLine={false} />
          <YAxis yAxisId="price" tick={{ fill: '#888780', fontSize: 10 }} axisLine={false} tickLine={false} width={52}
            tickFormatter={(v) => `$${v.toFixed(0)}`} />
          <YAxis yAxisId="hedge" orientation="right" tick={{ fill: '#ef9f27', fontSize: 10 }} axisLine={false}
            tickLine={false} width={36} tickFormatter={(v) => v.toFixed(2)} domain={[-1, 1]} />
          <Tooltip
            contentStyle={{ background: '#18181f', border: '1px solid #2a2a35', borderRadius: 8, fontSize: 11 }}
            labelStyle={{ color: '#888780' }}
          />
          <Legend wrapperStyle={{ fontSize: 10, color: '#888780', paddingTop: 4 }} />
          <Line yAxisId="price" type="monotone" dataKey="price" stroke="#7f77dd" strokeWidth={1.5}
            dot={false} name="Price" isAnimationActive={false} />
          <Line yAxisId="hedge" type="monotone" dataKey="hedge" stroke="#ef9f27" strokeWidth={1.5}
            dot={false} name="Hedge" isAnimationActive={false} strokeDasharray="4 2" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
