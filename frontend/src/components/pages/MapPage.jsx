import React, { useState, useMemo, useEffect, useCallback } from 'react';
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Safely extract latitude from either field name */
const getLat = (d) => {
  const v = d?.latitude ?? d?.lat ?? null;
  return v != null && !Number.isNaN(Number(v)) ? Number(v) : null;
};

/** Safely extract longitude from either field name */
const getLng = (d) => {
  const v = d?.longitude ?? d?.lng ?? null;
  return v != null && !Number.isNaN(Number(v)) ? Number(v) : null;
};

const getRiskColor = (risk) => {
  const r = String(risk || '').toLowerCase();
  if (r === 'high')   return { fill: '#E8604C', border: '#B91C1C', label: 'text-rose-700',   bg: 'bg-rose-50',   border2: 'border-rose-200' };
  if (r === 'medium') return { fill: '#F0A93B', border: '#D97706', label: 'text-amber-700',  bg: 'bg-amber-50',  border2: 'border-amber-200' };
  if (r === 'low')    return { fill: '#4FC98A', border: '#059669', label: 'text-emerald-700', bg: 'bg-emerald-50', border2: 'border-emerald-200' };
  return                     { fill: '#94A3B8', border: '#64748B', label: 'text-slate-600',  bg: 'bg-slate-100', border2: 'border-slate-300' };
};

const fmtCoord  = (v) => (v != null ? Number(v).toFixed(5) : '—');
const fmtConf   = (v) => { const n = Number(v || 0); return `${(n <= 1 ? n * 100 : n).toFixed(1)}%`; };
const fmtDepth  = (v) => (v != null ? `${Number(v).toFixed(1)} m` : '—');

// ─────────────────────────────────────────────────────────────────────────────
// Auto-fit map to visible markers whenever the filtered list changes
// ─────────────────────────────────────────────────────────────────────────────
function MapFitter({ detections }) {
  const map = useMap();
  useEffect(() => {
    const pts = detections
      .map((d) => [getLat(d), getLng(d)])
      .filter(([a, b]) => a != null && b != null);
    if (pts.length === 0) return;
    if (pts.length === 1) {
      map.setView(pts[0], Math.max(map.getZoom(), 10));
    } else {
      try { map.fitBounds(pts, { padding: [55, 55], maxZoom: 14 }); } catch { /* ignore */ }
    }
  }, [detections, map]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stat chip
// ─────────────────────────────────────────────────────────────────────────────
function StatChip({ label, value, colorClass = 'text-slate-900' }) {
  return (
    <div className="flex items-center gap-1.5 bg-white border border-slate-300 rounded-lg px-3 py-1.5 shadow-sm">
      <span className="text-[10.5px] font-mono text-slate-700 font-bold uppercase">{label}</span>
      <span className={`text-[13px] font-bold font-mono ${colorClass}`}>{value}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export function MapPage({ detections = [], surveys = [], activeSurvey = null, onNavigate }) {

  // ── filters ─────────────────────────────────────────────────────────────
  const [surveyFilter, setSurveyFilter]   = useState('all');
  const [riskFilter,   setRiskFilter]     = useState('all');
  const [classFilter,  setClassFilter]    = useState('all');

  // ── derive unique object classes from actual data ────────────────────────
  const objectClasses = useMemo(() => {
    const s = new Set();
    detections.forEach((d) => {
      const c = d?.object_class || d?.objectClass || d?.class;
      if (c) s.add(String(c));
    });
    return Array.from(s).sort();
  }, [detections]);

  // ── derive survey options from actual surveys prop ───────────────────────
  const surveyOptions = useMemo(() => {
    return surveys.map((s) => ({
      id:    s.id,
      label: s.name
        ? `SURV-${String(s.id).padStart(3, '0')} · ${s.name}`
        : `SURV-${String(s.id).padStart(3, '0')}`,
    }));
  }, [surveys]);

  // ── filter pipeline ──────────────────────────────────────────────────────
  const filteredDetections = useMemo(() => {
    return detections.filter((d) => {
      // Must have valid coordinates
      if (getLat(d) == null || getLng(d) == null) return false;

      // Survey filter
      if (surveyFilter !== 'all') {
        const sid = String(d?.survey_id ?? d?.surveyId ?? '');
        if (sid !== String(surveyFilter)) return false;
      }

      // Risk filter
      if (riskFilter !== 'all') {
        const r = String(d?.risk_level || d?.risk || '').toLowerCase();
        if (r !== riskFilter) return false;
      }

      // Object class filter
      if (classFilter !== 'all') {
        const c = String(d?.object_class || d?.objectClass || d?.class || '');
        if (c !== classFilter) return false;
      }

      return true;
    });
  }, [detections, surveyFilter, riskFilter, classFilter]);

  // ── summary counts ───────────────────────────────────────────────────────
  const summary = useMemo(() => {
    const high   = filteredDetections.filter((d) => String(d?.risk_level || d?.risk || '').toLowerCase() === 'high').length;
    const medium = filteredDetections.filter((d) => String(d?.risk_level || d?.risk || '').toLowerCase() === 'medium').length;
    const low    = filteredDetections.filter((d) => String(d?.risk_level || d?.risk || '').toLowerCase() === 'low').length;
    return { total: filteredDetections.length, high, medium, low };
  }, [filteredDetections]);

  // ── default map center ───────────────────────────────────────────────────
  const defaultCenter = useMemo(() => {
    const pts = detections
      .map((d) => [getLat(d), getLng(d)])
      .filter(([a, b]) => a != null && b != null);
    if (pts.length === 0) return [10.0, 76.0]; // Indian Ocean default
    const avgLat = pts.reduce((s, [a]) => s + a, 0) / pts.length;
    const avgLng = pts.reduce((s, [, b]) => s + b, 0) / pts.length;
    return [avgLat, avgLng];
  }, [detections]);
  const MapResizeHandler = () => {
  const map = useMap();

  useEffect(() => {
    const resizeMap = () => {
      map.invalidateSize();
    };

    resizeMap();

    const timer = setTimeout(resizeMap, 300);

    return () => clearTimeout(timer);
  }, [map]);

  return null;
};

  // ── select drop down helpers ─────────────────────────────────────────────
  const selectCls = 'bg-white border border-slate-200 rounded-lg text-[11.5px] font-mono text-slate-700 px-2.5 py-1.5 pr-7 shadow-sm focus:outline-none focus:ring-1 focus:ring-teal-400 appearance-none cursor-pointer';

  // ── no detections at all ─────────────────────────────────────────────────
  if (detections.length === 0) {
    return (
      <section className="animate-page-fade space-y-4 bg-[#EFF8FB] min-h-screen -m-6 p-6">
        <div className="border-b border-slate-100 pb-2">
          <h1 className="text-[18px] font-bold text-slate-800">Risk Map</h1>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-10 text-center max-w-md mx-auto my-8">
          <div className="w-11 h-11 rounded-full bg-teal-50 text-teal-600 mx-auto flex items-center justify-center mb-3">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="w-6 h-6">
              <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/>
              <circle cx="12" cy="9" r="2.5"/>
            </svg>
          </div>
          <h2 className="text-[14px] font-bold text-slate-800 mb-1.5">No Confirmed Detections to Display</h2>
          <p className="text-[12px] text-slate-500 mb-5">
            Accept detections in Human Verification to add them to the Risk Map.
          </p>
          <button onClick={() => onNavigate?.('review')} className="btn primary mx-auto text-[12px] py-1.5 px-4">
            Go to Human Verification →
          </button>
        </div>
      </section>
    );
  }

  // ── main render ───────────────────────────────────────────────────────────
  return (
    <section className="animate-page-fade bg-[#EFF8FB] min-h-screen -m-6 p-6 space-y-4">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-start justify-between gap-3 pb-2 border-b border-slate-200">
        <div>
          <p className="text-[16px] text-slate-900 font-medium">Confirmed detection locations — live from survey database</p>
        </div>
        {activeSurvey && (
          <span className="font-mono text-[11px] font-bold text-teal-800 bg-teal-50 border border-teal-300 px-2.5 py-1 rounded-lg self-start">
            Active: SURV-{String(activeSurvey.id).padStart(3, '0')}
          </span>
        )}
      </div>

      {/* ── Filter bar ────────────────────────────────────────────────────── */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm px-4 py-3 flex flex-wrap items-center gap-3">

        {/* Survey filter */}
        <div className="relative">
          <select value={surveyFilter} onChange={(e) => setSurveyFilter(e.target.value)} className={selectCls}>
            <option value="all">All Surveys</option>
            {surveyOptions.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
          <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 text-[10px]">▼</span>
        </div>

        {/* Risk Level filter */}
        <div className="relative">
          <select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)} className={selectCls}>
            <option value="all">All Risk Levels</option>
            <option value="high">High Risk</option>
            <option value="medium">Medium Risk</option>
            <option value="low">Low Risk</option>
          </select>
          <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 text-[10px]">▼</span>
        </div>

        {/* Object Class filter */}
        <div className="relative">
          <select value={classFilter} onChange={(e) => setClassFilter(e.target.value)} className={selectCls}>
            <option value="all">All Object Classes</option>
            {objectClasses.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 text-[10px]">▼</span>
        </div>

        {/* Reset */}
        {(surveyFilter !== 'all' || riskFilter !== 'all' || classFilter !== 'all') && (
          <button
            onClick={() => { setSurveyFilter('all'); setRiskFilter('all'); setClassFilter('all'); }}
            className="text-[11px] font-mono text-teal-600 hover:text-teal-800 underline underline-offset-2 transition-colors"
          >
            Reset filters
          </button>
        )}

        {/* Spacer + Stats */}
        <div className="flex-1" />
        <div className="flex flex-wrap gap-2">
          <StatChip label="Confirmed" value={summary.total} colorClass="text-teal-700" />
          <StatChip label="High"      value={summary.high}   colorClass="text-rose-600" />
          <StatChip label="Medium"    value={summary.medium} colorClass="text-amber-600" />
          <StatChip label="Low"       value={summary.low}    colorClass="text-emerald-600" />
        </div>
      </div>

      {/* ── Map area ──────────────────────────────────────────────────────── */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">

        {filteredDetections.length === 0 ? (
          /* Empty state — filters produced no results */
          <div className="h-[520px] flex flex-col items-center justify-center text-center p-8">
            <div className="w-10 h-10 rounded-full bg-slate-100 text-slate-400 mx-auto flex items-center justify-center mb-3">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="w-5 h-5">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
            </div>
            <p className="text-[13px] font-semibold text-slate-700 mb-1">No confirmed detections match the selected filters.</p>
            <p className="text-[11.5px] text-slate-400">Try adjusting the survey, risk level, or object class filter.</p>
          </div>
        ) : (
            <MapContainer
              center={[20.5937, 78.9629]}
              zoom={5}
              style={{ height: 'clamp(320px, 50vh, 600px)', width: '100%' }}
            >
              <MapResizeHandler />

            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              maxZoom={19}
            />

            {/* ── Auto-fit bounds to visible detections */}
            <MapFitter detections={filteredDetections} />

            {/* ── Detection markers */}
            {filteredDetections.map((det) => {
              const lat   = getLat(det);
              const lng   = getLng(det);
              if (lat == null || lng == null) return null;

              const risk   = String(det?.risk_level || det?.risk || '').toLowerCase();
              const colors = getRiskColor(risk);
              const detId  = det?.detection_id || det?.full_identifier || String(det?.id || '');
              const survCode = det?.survey_code || (det?.survey_id ? `SURV-${String(det.survey_id).padStart(3,'0')}` : '—');
              const objClass = det?.object_class || det?.objectClass || det?.class || 'Unknown';
              const conf   = fmtConf(det?.confidence);
              const depth  = fmtDepth(det?.depth ?? det?.depthMeters);
              const rScore = det?.risk_score ?? det?.riskScore;
              const status = String(det?.status || 'confirmed');

              // Radius scales with risk
              const radius = risk === 'high' ? 10 : risk === 'medium' ? 8 : 7;

              return (
                <CircleMarker
                  key={det?.id ?? detId}
                  center={[lat, lng]}
                  radius={radius}
                  pathOptions={{
                    color:       colors.border,
                    fillColor:   colors.fill,
                    fillOpacity: 0.88,
                    weight:      2.5,
                  }}
                >
                  <Popup
                    minWidth={220}
                    maxWidth={260}
                    className="aqua-popup"
                  >
                    <div style={{ fontFamily: 'system-ui, sans-serif' }}>
                      {/* Header */}
                      <div style={{
                        background: '#0f172a',
                        color: '#f0fdfa',
                        padding: '8px 10px',
                        margin: '-14px -20px 10px -20px',
                        borderRadius: '4px 4px 0 0',
                        fontSize: '11px',
                        fontFamily: 'monospace',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}>
                        <span style={{ fontWeight: 700, color: '#5eead4' }}>{detId}</span>
                        <span style={{ color: '#cbd5e1', fontWeight: 600 }}>{survCode}</span>
                      </div>

                      {/* Rows */}
                      {[
                        ['Object Class', objClass],
                        ['AI Confidence', conf],
                        ['Risk Score', rScore != null ? `${rScore} / 100` : '—'],
                        ['Risk Level', risk.toUpperCase()],
                        ['Latitude', fmtCoord(lat)],
                        ['Longitude', fmtCoord(lng)],
                        ['Depth', depth],
                        ['Status', status.toUpperCase()],
                      ].map(([label, value]) => (
                        <div key={label} style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '3.5px 0',
                          borderBottom: '1px solid #e2e8f0',
                          fontSize: '11.5px',
                        }}>
                          <span style={{ color: '#334155', fontWeight: 600 }}>{label}</span>
                          <span style={{ fontWeight: 700, color: '#091322', fontFamily: 'monospace' }}>{String(value)}</span>
                        </div>
                      ))}

                      {/* Risk badge */}
                      <div style={{
                        display: 'inline-block',
                        background: colors.fill,
                        color: '#fff',
                        borderRadius: '9999px',
                        padding: '1px 10px',
                        fontSize: '10px',
                        fontWeight: 700,
                        marginTop: '8px',
                        letterSpacing: '0.05em',
                      }}>
                        {risk.toUpperCase()} RISK
                      </div>

                      {/* Navigation button */}
                      <button
                        onClick={() => onNavigate?.('review', { selectedDetectionId: det?.id })}
                        style={{
                          display: 'block',
                          width: '100%',
                          marginTop: '10px',
                          padding: '5px 0',
                          background: '#0d9488',
                          color: '#fff',
                          border: 'none',
                          borderRadius: '6px',
                          fontSize: '11.5px',
                          fontWeight: 600,
                          cursor: 'pointer',
                          textAlign: 'center',
                        }}
                        onMouseEnter={(e) => { e.currentTarget.style.background = '#0f766e'; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = '#0d9488'; }}
                      >
                        View Detection →
                      </button>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>
        )}
      </div>

      {/* ── Legend + info bar ──────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Legend */}
        <div className="flex items-center gap-4 text-[12px] text-slate-800 font-semibold">
          <span className="font-mono text-[10.5px] text-slate-700 uppercase tracking-wider font-bold">Legend</span>
          {[
            { label: 'High Risk',   color: '#E8604C' },
            { label: 'Medium Risk', color: '#F0A93B' },
            { label: 'Low Risk',    color: '#4FC98A' },
          ].map(({ label, color }) => (
            <span key={label} className="inline-flex items-center gap-1.5 text-slate-800 font-bold">
              <span
                className="w-3 h-3 rounded-full inline-block border-2 flex-shrink-0"
                style={{ backgroundColor: color, borderColor: color }}
              />
              {label}
            </span>
          ))}
        </div>

        {/* Marker count */}
        <span className="font-mono text-[11px] text-slate-700 font-bold">
          {filteredDetections.length} marker{filteredDetections.length !== 1 ? 's' : ''} visible
          {detections.length !== filteredDetections.length ? ` (${detections.length} total confirmed)` : ''}
        </span>
      </div>
    </section>
  );
}