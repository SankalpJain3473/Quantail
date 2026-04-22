// src/components/TradeTable.tsx
import { Trade } from '../types';
interface Props { trades: Trade[] }

export function TradeTable({ trades }: Props) {
  return (
    <div className="bg-[#18181f] border border-[#2a2a35] rounded-lg overflow-hidden">
      <div className="px-3 py-2.5 border-b border-[#2a2a35] flex justify-between items-center">
        <span className="text-[10px] font-bold text-[#888780] uppercase tracking-wider">
          Trade Log — Full Audit Trail
        </span>
        <span className="text-[10px] text-[#534ab7] font-semibold">{trades.length} records</span>
      </div>
      <div className="overflow-x-auto max-h-56 overflow-y-auto">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-[#18181f]">
            <tr>
              {['#', 'Time', 'Source', 'Symbol', 'Side', 'Qty', 'Fill $',
                'Hedge Before→After', 'Error', 'Cost', 'CVaR', 'P&L', 'Reason'].map(h => (
                <th key={h} className="text-left px-2.5 py-2 text-[#888780] font-bold text-[10px] uppercase tracking-wider border-b border-[#2a2a35] whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trades.length === 0 ? (
              <tr>
                <td colSpan={13} className="text-center text-[#5f5e5a] py-8">
                  No trades yet — start a session
                </td>
              </tr>
            ) : (
              trades.map((t) => (
                <tr key={t.id} className="border-b border-[#1a1a22] hover:bg-[#1e1e28] transition-colors">
                  <td className="px-2.5 py-1.5 text-[#5f5e5a]">#{t.id}</td>
                  <td className="px-2.5 py-1.5 text-[#888780] whitespace-nowrap">
                    {new Date(t.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="px-2.5 py-1.5">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      t.source === 'live' || t.source === 'paper'
                        ? 'bg-[#0a2e1e] text-[#5dcaa5]'
                        : 'bg-[#1e1e28] text-[#888780]'
                    }`}>
                      {t.source.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-2.5 py-1.5 font-bold text-[#7f77dd]">{t.symbol}</td>
                  <td className="px-2.5 py-1.5">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      t.side === 'buy' ? 'bg-[#0a2e1e] text-[#5dcaa5]' : 'bg-[#3d1515] text-[#f09595]'
                    }`}>
                      {t.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-2.5 py-1.5">{t.qty}</td>
                  <td className="px-2.5 py-1.5">${t.fill_price.toFixed(2)}</td>
                  <td className="px-2.5 py-1.5 text-[#c2c0b6] whitespace-nowrap">
                    {t.hedge_before.toFixed(4)} → {t.hedge_after.toFixed(4)}
                  </td>
                  <td className={`px-2.5 py-1.5 ${Math.abs(t.hedging_error) > 0.05 ? 'text-[#f09595]' : ''}`}>
                    {t.hedging_error.toFixed(6)}
                  </td>
                  <td className="px-2.5 py-1.5 text-[#f09595]">${t.cost.toFixed(4)}</td>
                  <td className="px-2.5 py-1.5">{t.cvar.toFixed(4)}</td>
                  <td className={`px-2.5 py-1.5 font-semibold ${t.pnl >= 0 ? 'text-[#5dcaa5]' : 'text-[#f09595]'}`}>
                    {t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}
                  </td>
                  <td className="px-2.5 py-1.5 text-[#888780] text-[10px]">{t.reason}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
