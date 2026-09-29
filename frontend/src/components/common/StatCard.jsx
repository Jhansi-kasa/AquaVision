import React, { useEffect, useState } from 'react';

export const StatCard = ({ label, value, variant, pad = 1, delayIndex = 0 }) => {
  const [displayValue, setDisplayValue] = useState(
    String(0).padStart(pad, '0')
  );

  useEffect(() => {
    const target = typeof value === 'number' ? value : parseInt(value, 10);
    if (isNaN(target)) {
      setDisplayValue(String(value));
      return;
    }

    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (prefersReducedMotion) {
      setDisplayValue(String(target).padStart(pad, '0'));
      return;
    }

    const duration = 360;
    const startTime = performance.now();

    let animationFrameId;

    const step = (now) => {
      const progress = Math.min((now - startTime) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(ease * target);
      setDisplayValue(String(current).padStart(pad, '0'));

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(step);
      } else {
        setDisplayValue(String(target).padStart(pad, '0'));
      }
    };

    animationFrameId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animationFrameId);
  }, [value, pad]);

  const borderClass =
    variant === 'risk'
      ? 'border-l-coral-500'
      : variant === 'pending'
      ? 'border-l-amber-400'
      : variant === 'verified'
      ? 'border-l-sea-400'
      : 'border-l-teal-400';

  const delayStyle = {
    animationDelay: `${(delayIndex + 1) * 0.04}s`,
  };

  return (
    <div
      style={delayStyle}
      className={`bg-white border border-slate-200 border-l-[4px] ${borderClass} rounded-xl p-4 md:p-5 shadow-sm hover:shadow-md transition-shadow animate-card-entrance`}
    >
      <div className="text-[12px] text-slate-700 font-bold uppercase tracking-wider">{label}</div>
      <div className="font-mono text-[32px] font-bold mt-1.5 tracking-tight text-slate-900">
        {displayValue}
      </div>
    </div>
  );
};
