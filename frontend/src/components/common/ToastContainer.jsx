import React from 'react';
import { useToast } from '../../context/ToastContext';

export const ToastContainer = () => {
  const { toasts, removeToast } = useToast();

  const getIcon = (type) => {
    switch (type) {
      case 'success':
        return (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="w-[15px] h-[15px] flex-shrink-0 text-sea-400">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        );
      case 'warning':
        return (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="w-[15px] h-[15px] flex-shrink-0 text-amber-400">
            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        );
      case 'error':
        return (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="w-[15px] h-[15px] flex-shrink-0 text-coral-500">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
        );
      case 'info':
      default:
        return (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="w-[15px] h-[15px] flex-shrink-0 text-teal-400">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
        );
    }
  };

  const getBorderColor = (type) => {
    switch (type) {
      case 'success':
        return 'border-l-[3px] border-l-sea-400';
      case 'warning':
        return 'border-l-[3px] border-l-amber-400';
      case 'error':
        return 'border-l-[3px] border-l-coral-500';
      case 'info':
      default:
        return 'border-l-[3px] border-l-teal-400';
    }
  };

  return (
    <div className="fixed top-5 right-6 z-[9999] flex flex-col gap-2 pointer-events-none">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`pointer-events-auto min-w-[280px] max-w-[380px] p-[10px_14px] rounded-md bg-navy-900 border border-navy-700 shadow-[0_10px_28px_rgba(0,0,0,0.5)] flex items-center gap-2.5 text-xs text-foam-50 transition-all duration-200 ${getBorderColor(
            toast.type
          )} ${toast.exiting ? 'opacity-0 translate-x-4' : 'animate-toast-in'}`}
        >
          {getIcon(toast.type)}
          <div className="flex-1 leading-[1.35]">{toast.message}</div>
          <button
            onClick={() => removeToast(toast.id)}
            className="bg-transparent border-0 text-slate-500 hover:text-foam-50 cursor-pointer p-0 flex items-center justify-center text-sm font-semibold"
            aria-label="Dismiss"
          >
            &times;
          </button>
        </div>
      ))}
    </div>
  );
};
