export default function ScopePanel({ label, tag, boxes = [], scanning }) {
  return (
    <div className="bg-[#04141a] border border-slate-800 rounded-xl p-2.5 shadow-lg shadow-slate-900/10 transition-shadow duration-300 hover:shadow-teal-900/20">
      <div className="flex justify-between text-[9.5px] font-mono text-slate-400 mb-2 px-1">
        <span>{label}</span><span className="text-teal-400">{tag}</span>
      </div>
      <div className="relative h-40 rounded-lg overflow-hidden bg-[#031015] ring-1 ring-teal-500/10">
        <div className="absolute inset-0 opacity-30" style={{ backgroundImage: 'repeating-linear-gradient(100deg, rgba(45,212,196,.12) 0px, transparent 3px, transparent 9px)' }}></div>
        {scanning && <div className="absolute inset-x-0 h-10 bg-gradient-to-b from-transparent via-teal-400/15 to-transparent animate-scan"></div>}
        {boxes.map((b, i) => (
          <div
            key={i}
            className="absolute border-2 rounded-sm animate-popIn"
            style={{ left: b.x + '%', top: b.y + '%', width: b.w + '%', height: b.h + '%', borderColor: b.color, animationDelay: i * 80 + 'ms' }}
          >
            <span className="absolute -top-4 left-0 text-[8px] font-mono px-1 rounded-sm whitespace-nowrap text-slate-900" style={{ background: b.color }}>
              {b.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}