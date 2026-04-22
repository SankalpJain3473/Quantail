// src/components/KPICard.tsx
interface KPICardProps {
  title: string;
  value: string;
  sub: string;
  color: 'green' | 'red' | 'purple' | 'amber';
  bar: { value: number; color: string } | null;
}

const colorMap = {
  green: 'text-[#5dcaa5]',
  red: 'text-[#f09595]',
  purple: 'text-[#7f77dd]',
  amber: 'text-[#ef9f27]',
};

const barColorMap: Record<string, string> = {
  green: 'bg-[#0f6e56]',
  red: 'bg-[#a32d2d]',
  purple: 'bg-[#534ab7]',
  amber: 'bg-[#854f0b]',
};

export function KPICard({ title, value, sub, color, bar }: KPICardProps) {
  return (
    <div className="bg-[#18181f] border border-[#2a2a35] rounded-lg p-3">
      <div className="text-[10px] font-bold text-[#888780] uppercase tracking-wider mb-1.5">{title}</div>
      <div className={`text-2xl font-bold leading-none mb-1 ${colorMap[color]}`}>{value}</div>
      <div className="text-[11px] text-[#888780]">{sub}</div>
      {bar && (
        <div className="mt-2 bg-[#0f0f13] rounded-full h-1.5 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${barColorMap[bar.color] || 'bg-[#534ab7]'}`}
            style={{ width: `${Math.max(0, Math.min(100, bar.value))}%` }}
          />
        </div>
      )}
    </div>
  );
}
