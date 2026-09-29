import { useCallback, useState } from 'react';
import { Icons } from './Icons.jsx';

let toastId = 0;

export function useToasts() {
  const [toasts, setToasts] = useState([]);
  const push = useCallback((type, message) => {
    const id = ++toastId;
    setToasts((t) => [...t, { id, type, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3600);
  }, []);
  return [toasts, push];
}

const TOAST_STYLE = {
  success: { border: 'border-l-emerald-500', icon: Icons.check, color: 'text-emerald-600', chip: 'bg-emerald-50' },
  info: { border: 'border-l-teal-500', icon: Icons.info, color: 'text-teal-600', chip: 'bg-teal-50' },
  warning: { border: 'border-l-amber-500', icon: Icons.alert, color: 'text-amber-600', chip: 'bg-amber-50' },
  error: { border: 'border-l-rose-500', icon: Icons.x, color: 'text-rose-600', chip: 'bg-rose-50' },
};

export default function Toasts({ toasts }) {
  return (
    <div className="fixed top-5 right-5 z-[9999] flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => {
        const s = TOAST_STYLE[t.type] || TOAST_STYLE.info;
        const Icn = s.icon;
        return (
          <div key={t.id} className={`pointer-events-auto animate-slideIn min-w-[280px] max-w-[380px] bg-white border border-slate-200 border-l-4 ${s.border} rounded-lg shadow-lg px-3.5 py-3 flex items-center gap-3 text-[13px] text-slate-700`}>
            <span className={`w-7 h-7 rounded-full ${s.chip} flex items-center justify-center flex-shrink-0`}>
              <Icn className={`w-4 h-4 ${s.color}`} />
            </span>
            <div className="flex-1 leading-snug">{t.message}</div>
          </div>
        );
      })}
    </div>
  );
}