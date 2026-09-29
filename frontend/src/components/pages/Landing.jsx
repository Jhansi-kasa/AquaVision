// src/components/pages/Landing.jsx
//
// Real side-scan sonar image (AI4Shipwrecks · Monrovia) shown as a scrolling
// "waterfall" live feed, with labeled detection boxes that scroll with the seabed.
//
// SETUP
//  1. Put sonar.jpg in  public/sonar.jpg
//  2. Everything you may want to change is in the Config block below.
//
// The image is a recorded survey replayed on a loop. It has been prepared so the
// bottom blends into the top, so the scroll has no visible seam.

import { useState } from 'react';

/* ------------------------------------------------------------------ */
/* Config: edit this block only                                        */
/* ------------------------------------------------------------------ */

const SONAR_IMAGE = {
  src: '/sonar.jpg',
  width: 1728,
  height: 1532,
  credit: 'Recorded survey replayed as a live feed · AI4Shipwrecks (Monrovia)',
};

const SCROLL_SECONDS = 45;   // time for one full pass of the image (higher = slower)
const VIEW_ASPECT = '4 / 3'; // shape of the visible window

// Order must match the class IDs used in DETECTIONS.
const CLASS_NAMES = ['Shipwreck', 'Debris', 'Ghost net', 'Fishing gear'];

// Color per class ID (border + label chip).
const CLASS_COLORS = ['#fbbf24', '#60a5fa', '#f87171', '#c084fc'];

// YOLO format, relative to the full image: cls, xc, yc, w, h (all 0..1).
// Add conf (0..1) to a box to show a percentage in its label.
const DETECTIONS = [
  { cls: 0, xc: 0.7321, yc: 0.5245, w: 0.272,  h: 0.3636 }, // shipwreck hull + wreckage
  { cls: 1, xc: 0.298,  yc: 0.6854, w: 0.0266, h: 0.0287 }, // small debris, left channel
  { cls: 1, xc: 0.8024, yc: 0.9683, w: 0.0249, h: 0.0372 }, // small debris, right channel
];

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

// YOLO (center-based, normalized) -> CSS percentages for an absolutely positioned box.
const yoloToCss = ({ xc, yc, w, h }) => ({
  left: `${(xc - w / 2) * 100}%`,
  top: `${(yc - h / 2) * 100}%`,
  width: `${w * 100}%`,
  height: `${h * 100}%`,
});

const labelText = (d) => {
  const name = CLASS_NAMES[d.cls] ?? `Class ${d.cls}`;
  return d.conf != null ? `${name} ${Math.round(d.conf * 100)}%` : name;
};

/* ------------------------------------------------------------------ */
/* UI pieces                                                           */
/* ------------------------------------------------------------------ */

const Btn = ({ children, className = '', onClick }) => (
  <button
    onClick={onClick}
    className={`inline-flex items-center gap-2 font-semibold text-white bg-teal-600 hover:bg-teal-700 active:bg-teal-800 rounded-lg shadow-sm transition-colors ${className}`}
  >
    {children}
  </button>
);

const Arrow = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M5 12h14M13 6l6 6-6 6" />
  </svg>
);

const DetectionBoxes = () =>
  DETECTIONS.map((d, i) => {
    const color = CLASS_COLORS[d.cls % CLASS_COLORS.length];
    return (
      <div
        key={i}
        className="absolute border-2 rounded-sm pointer-events-none"
        style={{ ...yoloToCss(d), borderColor: color }}
      >
        <span
          className="absolute -top-5 left-0 text-[9px] sm:text-[10px] font-mono text-slate-900 px-1 rounded-sm whitespace-nowrap"
          style={{ backgroundColor: color }}
        >
          {labelText(d)}
        </span>
      </div>
    );
  });

// One copy of the image plus its boxes. Two copies are stacked so the loop is endless.
const Tile = ({ style, onError }) => (
  <div className="absolute left-0 w-full h-full" style={style}>
    <img
      src={SONAR_IMAGE.src}
      alt="Side-scan sonar waterfall showing a shipwreck on the seabed"
      className="absolute inset-0 w-full h-full object-fill select-none"
      draggable={false}
      onError={onError}
    />
    <DetectionBoxes />
  </div>
);

const SonarScanner = () => {
  const [failed, setFailed] = useState(false);

  return (
    <div className="bg-slate-900 rounded-2xl p-3 sm:p-4 shadow-2xl shadow-slate-300/50">
      {/* Keyframes live here so the component is self-contained */}
      <style>{`
        @keyframes av-sonar-scroll { from { transform: translateY(0); } to { transform: translateY(100%); } }
        .av-sonar-track { animation: av-sonar-scroll ${SCROLL_SECONDS}s linear infinite; will-change: transform; }
        @media (prefers-reduced-motion: reduce) { .av-sonar-track { animation: none; } }
      `}</style>

      <div className="flex items-center justify-between text-[10px] sm:text-[10.5px] font-mono text-slate-400 mb-2 px-1">
        <span>DETECTIONS · LIVE</span>
        <span className="text-teal-400">SCANNING…</span>
      </div>

      <div
        className="relative w-full rounded-lg overflow-hidden bg-[#04141a]"
        style={{ aspectRatio: VIEW_ASPECT }}
      >
        {failed ? (
          <div className="absolute inset-0 flex items-center justify-center text-[11px] font-mono text-slate-500">
            Sonar image unavailable
          </div>
        ) : (
          /* Track = one image tall. Copy A sits in place, copy B sits directly above it.
             Sliding the track down by 100% puts B exactly where A started: seamless. */
          <div
            className="av-sonar-track absolute left-0 top-0 w-full"
            style={{ aspectRatio: `${SONAR_IMAGE.width} / ${SONAR_IMAGE.height}` }}
          >
            <Tile style={{ top: 0 }} onError={() => setFailed(true)} />
            <Tile style={{ top: '-100%' }} />
          </div>
        )}

        {/* Scan-line texture */}
        <div
          className="absolute inset-0 opacity-25 pointer-events-none"
          style={{ backgroundImage: 'repeating-linear-gradient(0deg, rgba(45,212,196,.15) 0px, transparent 2px, transparent 4px)' }}
        ></div>

        {/* Incoming-data edge: new pings enter at the top */}
        <div className="absolute inset-x-0 top-0 h-10 bg-gradient-to-b from-teal-400/25 to-transparent pointer-events-none"></div>
        <div className="absolute inset-x-0 top-0 h-px bg-teal-300/70 pointer-events-none"></div>
      </div>

      <div className="text-[10.5px] sm:text-[11px] text-slate-500 font-mono mt-2 px-1">
        {SONAR_IMAGE.credit} — tap Get started for the full console.
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export const Landing = ({ onEnter }) => {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-white to-slate-50 relative overflow-hidden">
      {/* Top bar: brand + Get started (mobile only) */}
      <header className="md:hidden sticky top-0 z-20 flex items-center justify-between px-4 py-3 bg-white/80 backdrop-blur border-b border-slate-100">
        <span className="text-lg font-extrabold tracking-tight text-slate-900">AquaVision</span>
        <Btn className="px-4 py-2 text-[13px]" onClick={onEnter}>
          Get started <Arrow />
        </Btn>
      </header>

      <main className="flex-1 flex items-center">
        <div className="max-w-6xl mx-auto w-full px-5 sm:px-8 py-8 md:py-16 grid md:grid-cols-2 gap-8 md:gap-14 items-center">
          {/* Text */}
          <div className="text-center md:text-left">
            <span className="inline-flex items-center gap-2 text-[11px] font-mono text-teal-700 bg-teal-50 border border-teal-200 px-3 py-1.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse"></span> Live sonar feed · demo
            </span>

            <h1 className="text-5xl sm:text-6xl font-extrabold leading-none tracking-tight text-slate-900 mt-5">
              AquaVision
            </h1>

            <h2 className="text-xl sm:text-2xl font-semibold leading-snug text-teal-700 mt-3">
              Underwater Marine debris and anomalies detection system.
            </h2>

            <p className="text-slate-500 text-[15px] md:text-base leading-relaxed mt-4 max-w-md mx-auto md:mx-0">
              AquaVision scans side-scan sonar images, detects ghost nets and anomalies, and turns them into a reviewed, prioritized cleanup plan.
            </p>

            {/* Desktop CTA (mobile uses the top bar button) */}
            <div className="mt-7 hidden md:flex items-center gap-4">
              <Btn className="px-6 py-3 text-[14px]" onClick={onEnter}>
                Get started <Arrow />
              </Btn>
            </div>
          </div>

          {/* Sonar scanner */}
          <SonarScanner />
        </div>
      </main>
    </div>
  );
};

export default Landing;