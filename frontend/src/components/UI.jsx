import { RISK_STYLE } from '../data.js';

export const Btn = ({ children, variant = 'default', className = '', ...rest }) => {
  const base = 'inline-flex items-center justify-center gap-2 rounded-lg text-sm font-semibold px-4 py-2.5 transition-all duration-200 active:scale-[.96] disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:shadow-none';
  const variants = {
    // Brand CTA — teal/cyan gradient with lift on hover
    primary: 'bg-gradient-to-br from-cyan-500 to-teal-500 text-white shadow-md shadow-teal-500/25 hover:shadow-lg hover:shadow-teal-500/35 hover:-translate-y-0.5 hover:from-cyan-400 hover:to-teal-400',
    // Secondary strong action — violet/indigo gradient, used for routing & workflow moves
    secondary: 'bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-md shadow-indigo-500/25 hover:shadow-lg hover:shadow-indigo-500/35 hover:-translate-y-0.5 hover:from-indigo-400 hover:to-violet-400',
    // Neutral
    default: 'bg-white text-slate-600 border border-slate-200 hover:border-teal-300 hover:text-teal-700 hover:bg-teal-50/60 hover:-translate-y-0.5 hover:shadow-sm',
    // Positive / confirm
    confirm: 'bg-gradient-to-br from-emerald-500 to-emerald-600 text-white shadow-md shadow-emerald-500/25 hover:shadow-lg hover:shadow-emerald-500/35 hover:-translate-y-0.5 hover:from-emerald-400 hover:to-emerald-500',
    // Destructive / reject
    reject: 'bg-white text-rose-500 border border-rose-200 hover:bg-rose-50 hover:border-rose-300 hover:-translate-y-0.5 hover:shadow-sm hover:shadow-rose-200/60',
    // Caution / flag-for-review actions
    warn: 'bg-gradient-to-br from-amber-500 to-orange-500 text-white shadow-md shadow-amber-500/25 hover:shadow-lg hover:shadow-amber-500/35 hover:-translate-y-0.5 hover:from-amber-400 hover:to-orange-400',
    // Data export — sky
    sky: 'bg-gradient-to-br from-sky-500 to-blue-500 text-white shadow-md shadow-sky-500/25 hover:shadow-lg hover:shadow-sky-500/35 hover:-translate-y-0.5',
    ghost: 'text-slate-500 hover:text-teal-600 hover:bg-teal-50',
  };
  return <button className={`${base} ${variants[variant]} ${className}`} {...rest}>{children}</button>;
};

// accent: optional left-edge color bar tying the panel to a category (e.g. 'teal', 'rose', 'amber', 'indigo')
const ACCENT_BAR = {
  teal: 'before:bg-gradient-to-b before:from-cyan-400 before:to-teal-500',
  rose: 'before:bg-gradient-to-b before:from-rose-400 before:to-rose-500',
  amber: 'before:bg-gradient-to-b before:from-amber-400 before:to-orange-500',
  emerald: 'before:bg-gradient-to-b before:from-emerald-400 before:to-emerald-500',
  indigo: 'before:bg-gradient-to-b before:from-indigo-400 before:to-violet-500',
};

export const Panel = ({ title, tag, className = '', accent, hover = false, children }) => (
  <div
    className={`relative bg-white border border-slate-200 rounded-xl p-5 shadow-sm overflow-hidden
      ${accent ? `pl-6 before:absolute before:left-0 before:top-0 before:h-full before:w-1 ${ACCENT_BAR[accent] || ''}` : ''}
      ${hover ? 'transition-all duration-300 hover:shadow-lg hover:-translate-y-0.5 hover:border-teal-200' : ''}
      ${className}`}
  >
    {title && (
      <div className="flex items-center justify-between mb-4">
        <div className="text-[13px] font-semibold text-slate-800">{title}</div>
        {tag && <span className="text-[11px] font-mono text-teal-700 bg-teal-50 border border-teal-100 px-2 py-0.5 rounded-full">{tag}</span>}
      </div>
    )}
    {children}
  </div>
);

export const KV = ({ k, v }) => (
  <div className="flex items-center justify-between py-2 border-b border-dashed border-slate-100 last:border-0 text-[13px]">
    <span className="text-slate-500">{k}</span>
    <span className="font-mono font-medium text-slate-800">{v}</span>
  </div>
);

export const Badge = ({ risk }) => {
  const s = RISK_STYLE[risk] || RISK_STYLE.unknown;
  return (
    <span className={`inline-flex items-center gap-1.5 font-mono text-[10.5px] font-bold px-2.5 py-1 rounded-full ${s.bg} ${s.text} ring-1 ring-inset ${s.ring}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot} ${risk === 'high' ? 'animate-pulse' : ''}`}></span>
      {risk.toUpperCase()}
    </span>
  );
};

export const Pill = ({ children, tone = 'default' }) => {
  const tones = {
    default: 'border-slate-200 text-slate-500 bg-white',
    good: 'border-emerald-200 text-emerald-600 bg-emerald-50',
    info: 'border-teal-200 text-teal-700 bg-teal-50',
    warn: 'border-amber-200 text-amber-600 bg-amber-50',
  };
  return <span className={`text-[11px] font-medium px-2.5 py-1 rounded-full border ${tones[tone]}`}>{children}</span>;
};