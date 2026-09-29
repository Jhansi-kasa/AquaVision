import { useEffect, useState } from 'react';

// Maps the accent bar color to a matching gradient icon badge + glow, so each
// stat reads as its own color story instead of one flat grey icon tile.
const ACCENT_THEME = {
  'bg-teal-500': { icon: 'bg-gradient-to-br from-cyan-400 to-teal-500 text-white', glow: 'group-hover:shadow-teal-200/70' },
  'bg-rose-500': { icon: 'bg-gradient-to-br from-rose-400 to-rose-500 text-white', glow: 'group-hover:shadow-rose-200/70' },
  'bg-amber-500': { icon: 'bg-gradient-to-br from-amber-400 to-orange-500 text-white', glow: 'group-hover:shadow-amber-200/70' },
  'bg-emerald-500': { icon: 'bg-gradient-to-br from-emerald-400 to-emerald-500 text-white', glow: 'group-hover:shadow-emerald-200/70' },
};

export default function StatCard({ label, value, accent, delay, icon: Icn }) {
  const [display, setDisplay] = useState(0);
  const theme = ACCENT_THEME[accent] || { icon: 'bg-slate-100 text-slate-500', glow: '' };

  useEffect(() => {
    let raf, start;
    const dur = 700;
    const tick = (t) => {
      if (!start) start = t;
      const p = Math.min((t - start) / dur, 1);
      setDisplay(Math.round((1 - Math.pow(1 - p, 3)) * value));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    const to = setTimeout(() => (raf = requestAnimationFrame(tick)), delay);
    return () => { clearTimeout(to); cancelAnimationFrame(raf); };
  }, [value, delay]);

  return (
    <div
      className={`group relative bg-white border border-slate-200 rounded-xl p-5 overflow-hidden shadow-sm hover:shadow-xl ${theme.glow} hover:-translate-y-1 transition-all duration-300 animate-fadeUp`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className={`absolute left-0 top-0 h-full w-1 ${accent}`}></div>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[12px] text-slate-500 font-medium">{label}</div>
          <div className="mono text-3xl font-bold text-slate-800 mt-2 tabular-nums">{display}</div>
        </div>
        <div className={`w-10 h-10 rounded-lg ${theme.icon} flex items-center justify-center shadow-sm transition-transform duration-300 group-hover:scale-110 group-hover:rotate-3`}>
          <Icn className="w-4.5 h-4.5" />
        </div>
      </div>
    </div>
  );
}