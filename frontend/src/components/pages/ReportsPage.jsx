import React, { useState, useMemo, useEffect } from 'react';
import { useToast } from '../../context/ToastContext';
import api from '../../services/api';

/* ─────────────────────────────────────────────────────────────
   Formatting helpers
───────────────────────────────────────────────────────────── */
const fmtSurvCode = (id) => `SURV-${String(id).padStart(3, '0')}`;

const fmtDetCode = (d) =>
  d.detection_id ||
  d.full_identifier ||
  (d.survey_detection_index != null
    ? `DET-${String(d.survey_detection_index).padStart(3, '0')}`
    : `DET-${String(d.id).padStart(3, '0')}`);

const fmtConfidence = (raw) => {
  const n = Number(raw);
  if (Number.isNaN(n)) return '—';
  const pct = n <= 1 ? n * 100 : n;
  return `${pct.toFixed(1)}%`;
};

const fmtRiskScore = (raw) => {
  const n = Number(raw);
  return Number.isNaN(n) ? '—' : n.toFixed(1);
};

const fmtCoords = (lat, lng) => {
  const la = Number(lat);
  const lo = Number(lng);
  if (Number.isNaN(la) || Number.isNaN(lo)) return '—';
  return `${la.toFixed(4)}, ${lo.toFixed(4)}`;
};

const fmtDepth = (d) =>
  d != null && !Number.isNaN(Number(d)) ? `${Number(d).toFixed(1)} m` : '—';

const riskLevel = (d) => {
  const explicit = String(d.risk_level || d.risk || '').toUpperCase();
  if (['HIGH', 'MEDIUM', 'LOW'].includes(explicit)) return explicit;
  const score = Number(d.risk_score || d.riskScore || 0);
  if (score >= 70) return 'HIGH';
  if (score >= 40) return 'MEDIUM';
  return 'LOW';
};

/* ─────────────────────────────────────────────────────────────
   Badge style maps — pill-shaped, soft colored borders (no black)
───────────────────────────────────────────────────────────── */
const RISK_BADGE = {
  HIGH:   'bg-rose-50 text-rose-700 border border-rose-200',
  MEDIUM: 'bg-amber-50 text-amber-700 border border-amber-200',
  LOW:    'bg-emerald-50 text-emerald-700 border border-emerald-200',
};

const STATUS_BADGE = {
  pending:   'bg-amber-50 text-amber-700 border border-amber-200',
  flagged:   'bg-orange-50 text-orange-700 border border-orange-200',
  confirmed: 'bg-teal-50 text-teal-700 border border-teal-200',
  accepted:  'bg-teal-50 text-teal-700 border border-teal-200',
  verified:  'bg-emerald-50 text-emerald-700 border border-emerald-200',
  cleaned:   'bg-indigo-50 text-indigo-700 border border-indigo-200',
  rejected:  'bg-rose-50 text-rose-700 border border-rose-200',
};

const statusBadgeCls = (s) =>
  STATUS_BADGE[String(s).toLowerCase()] || STATUS_BADGE.pending;

/* Cleanup / verification result → badge */
const VERIFY_BADGE = {
  cleared:              'bg-emerald-50 text-emerald-700 border border-emerald-200',
  potentially_removed:  'bg-emerald-50 text-emerald-700 border border-emerald-200',
  not_cleared:          'bg-rose-50 text-rose-700 border border-rose-200',
  still_present:        'bg-rose-50 text-rose-700 border border-rose-200',
  inconclusive:         'bg-amber-50 text-amber-700 border border-amber-200',
  not_verified:         'bg-rose-50 text-rose-700 border border-rose-200',
  pending:              'bg-rose-50 text-rose-700 border border-rose-200',
};

const VERIFY_LABEL = {
  cleared:             'Verified',
  potentially_removed: 'Verified',
  not_cleared:         'Not Cleared',
  still_present:       'Still Present',
  inconclusive:        'Inconclusive',
  not_verified:        'Not Verified',
  pending:              'Pending',
};

const verifyBadge = (result) => {
  const key = String(result || 'pending').toLowerCase().replace(/ /g, '_');
  return {
    cls:   VERIFY_BADGE[key] || VERIFY_BADGE.pending,
    label: VERIFY_LABEL[key] || String(result || 'Pending').replace(/_/g, ' '),
  };
};

/* ─────────────────────────────────────────────────────────────
   Shared filter/select style
───────────────────────────────────────────────────────────── */
const SEL_CLS =
  'h-9 border border-slate-200 bg-white text-slate-700 rounded-lg px-3 text-[12px] font-medium ' +
  'focus:outline-none focus:border-teal-400 focus:ring-2 focus:ring-teal-100 shadow-sm transition-colors ' +
  'hover:border-slate-300 appearance-none cursor-pointer';

const INPUT_CLS =
  'h-9 border border-slate-200 bg-white text-slate-700 rounded-lg px-3 text-[12px] ' +
  'focus:outline-none focus:border-teal-400 focus:ring-2 focus:ring-teal-100 shadow-sm transition-colors ' +
  'hover:border-slate-300 placeholder:text-slate-400';

/* ─────────────────────────────────────────────────────────────
   Chevron icon for selects
───────────────────────────────────────────────────────────── */
function SelectWrapper({ children, className = '' }) {
  return (
    <div className={`relative ${className}`}>
      {children}
      <svg
        className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500"
        viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
      >
        <polyline points="6 9 12 15 18 9" />
      </svg>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   REPORTS PAGE
═══════════════════════════════════════════════════════════ */
export const ReportsPage = ({
  detections = [],
  surveys    = [],
  currentSurveyId,
  onSelectSurvey,
}) => {
  const { showToast } = useToast();

  /* ── Survey selector ─────────────────────────────────── */
  const [selectedSurveyId, setSelectedSurveyId] = useState(() => {
    if (currentSurveyId) return String(currentSurveyId);
    if (surveys.length > 0) return String(surveys[0].id);
    return '';
  });

  useEffect(() => {
    if (currentSurveyId) setSelectedSurveyId(String(currentSurveyId));
  }, [currentSurveyId]);

  /* ── Verifications (cleanup status) ─────────────────── */
  const [verifications, setVerifications] = useState([]);
  useEffect(() => {
    api.getVerifications().then((data) => {
      if (Array.isArray(data)) setVerifications(data);
    }).catch(() => {});
  }, []);

  // Map detection_id → verification record
  const verifyMap = useMemo(() => {
    const m = {};
    verifications.forEach((v) => { m[String(v.detection_id)] = v; });
    return m;
  }, [verifications]);

  /* ── Survey change ───────────────────────────────────── */
  const handleSurveyChange = (e) => {
    const val = e.target.value;
    setSelectedSurveyId(val);
    setFilterClass('');
    setFilterRisk('');
    setFilterStatus('');
    setFilterVerify('');
    setFilterSearch('');
    if (val && onSelectSurvey) onSelectSurvey(Number(val));
  };

  /* ── Active survey ───────────────────────────────────── */
  const activeSurvey = useMemo(
    () => surveys.find((s) => String(s.id) === selectedSurveyId) || null,
    [surveys, selectedSurveyId],
  );

  /* ── Scoped detections ───────────────────────────────── */
  const scopedDetections = useMemo(() => {
    if (!activeSurvey) return [];
    return detections.filter(
      (d) => String(d.survey_id ?? d.surveyId) === String(activeSurvey.id),
    );
  }, [detections, activeSurvey]);

  /* ── Statistics ──────────────────────────────────────── */
  const stats = useMemo(() => {
    const total     = scopedDetections.length;
    const high      = scopedDetections.filter((d) => riskLevel(d) === 'HIGH').length;
    const medium    = scopedDetections.filter((d) => riskLevel(d) === 'MEDIUM').length;
    const low       = scopedDetections.filter((d) => riskLevel(d) === 'LOW').length;
    const confirmed = scopedDetections.filter((d) =>
      ['confirmed', 'verified', 'accepted', 'cleaned'].includes(
        String(d.status).toLowerCase(),
      ),
    ).length;
    const cleaned = scopedDetections.filter(
      (d) => String(d.status).toLowerCase() === 'cleaned',
    ).length;
    const verified = scopedDetections.filter((d) => verifyMap[String(d.id)]).length;
    return { total, high, medium, low, confirmed, cleaned, verified };
  }, [scopedDetections, verifyMap]);

  /* ── Filter state ────────────────────────────────────── */
  const [filterClass,  setFilterClass]  = useState('');
  const [filterRisk,   setFilterRisk]   = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [filterVerify, setFilterVerify] = useState('');
  const [filterSearch, setFilterSearch] = useState('');

  const objectClasses = useMemo(
    () =>
      [...new Set(
        scopedDetections
          .map((d) => d.object_class || d.objectClass || 'Unknown')
          .filter(Boolean),
      )].sort(),
    [scopedDetections],
  );

  const filteredDetections = useMemo(() => {
    let list = scopedDetections;
    if (filterClass)  list = list.filter((d) => (d.object_class || d.objectClass || '') === filterClass);
    if (filterRisk)   list = list.filter((d) => riskLevel(d) === filterRisk);
    if (filterStatus) list = list.filter((d) => String(d.status || 'pending').toLowerCase() === filterStatus);
    if (filterVerify === 'verified')   list = list.filter((d) =>  verifyMap[String(d.id)]);
    if (filterVerify === 'unverified') list = list.filter((d) => !verifyMap[String(d.id)]);
    if (filterSearch) {
      const q = filterSearch.toLowerCase();
      list = list.filter((d) => {
        const code = fmtDetCode(d).toLowerCase();
        const cls  = (d.object_class || d.objectClass || '').toLowerCase();
        return code.includes(q) || cls.includes(q);
      });
    }
    return list;
  }, [scopedDetections, filterClass, filterRisk, filterStatus, filterVerify, filterSearch, verifyMap]);

  const hasActiveFilter = filterClass || filterRisk || filterStatus || filterVerify || filterSearch;

  /* ── Export handlers ─────────────────────────────────── */
  const handleExport = (format) => {
    if (!scopedDetections.length) {
      showToast({ type: 'warning', message: 'No detections to export for this survey.' });
      return;
    }
    const survCode = activeSurvey ? fmtSurvCode(activeSurvey.id) : 'UNKNOWN';

    if (format === 'pdf') {
      showToast({ type: 'info', message: 'Opening print dialog…' });
      setTimeout(() => window.print(), 350);
      return;
    }

    if (format === 'csv') {
      const header =
        'Survey_Code,Detection_ID,Object_Class,Confidence,Risk_Level,Risk_Score,' +
        'Latitude,Longitude,Depth_m,Status,Cleanup_Verification,Verify_Confidence\n';
      const rows = scopedDetections.map((d) => {
        const conf   = Number(d.confidence ?? 0);
        const pct    = (conf <= 1 ? conf * 100 : conf).toFixed(1);
        const vr     = verifyMap[String(d.id)];
        const vrRes  = vr ? (vr.result || '—') : 'Pending';
        const vrConf = vr ? `${((vr.confidence || 0) * 100).toFixed(0)}%` : '—';
        return `"${survCode}","${fmtDetCode(d)}","${d.object_class || 'Unknown'}","${pct}%","${riskLevel(d)}",${fmtRiskScore(d.risk_score || d.riskScore)},${d.latitude ?? d.lat ?? ''},${d.longitude ?? d.lng ?? ''},${d.depth ?? ''},"${d.status || 'pending'}","${vrRes}","${vrConf}"`;
      });
      const blob = new Blob([header + rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `aquavision_${survCode}_report.csv`;
      a.click();
      URL.revokeObjectURL(url);
      showToast({ type: 'success', message: 'Exported as CSV.' });
      return;
    }

    if (format === 'json') {
      const payload = {
        survey:      activeSurvey,
        survey_code: survCode,
        export_date: new Date().toISOString(),
        statistics:  stats,
        detections:  scopedDetections.map((d) => ({
          ...d,
          cleanup_verification: verifyMap[String(d.id)] || null,
        })),
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `aquavision_${survCode}_report.json`;
      a.click();
      URL.revokeObjectURL(url);
      showToast({ type: 'success', message: 'Exported as JSON.' });
    }
  };

  /* ── Empty state guards ──────────────────────────────── */
  if (surveys.length === 0) {
    return (
      <section className="animate-page-fade bg-[#EFF8FB] min-h-screen -m-6 p-6 flex flex-col">
        <ToolbarHeader surveys={surveys} selectedSurveyId={selectedSurveyId} onSurveyChange={handleSurveyChange} />
        <div className="flex-1 flex items-center justify-center">
          <EmptyState icon="📋" message="No survey reports available." />
        </div>
      </section>
    );
  }

  if (!activeSurvey) {
    return (
      <section className="animate-page-fade bg-[#EFF8FB] min-h-screen -m-6 p-6 flex flex-col space-y-4">
        <ToolbarHeader surveys={surveys} selectedSurveyId={selectedSurveyId} onSurveyChange={handleSurveyChange} />
        <div className="flex-1 flex items-center justify-center">
          <EmptyState icon="🔍" message="Select a survey to view its report." />
        </div>
      </section>
    );
  }

  const survCode = fmtSurvCode(activeSurvey.id);

  /* ═══════════════════════════════════════════════════════
     RENDER
  ═══════════════════════════════════════════════════════ */
  return (
    <section className="animate-page-fade space-y-4 bg-[#EFF8FB] min-h-screen -m-6 p-6">

      {/* ── Toolbar ──────────────────────────────────────── */}
      <ToolbarHeader
        surveys={surveys}
        selectedSurveyId={selectedSurveyId}
        onSurveyChange={handleSurveyChange}
      />

      {/* ═══ Survey Report Card ══════════════════════════ */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">

        {/* Survey identity */}
        <div className="flex flex-wrap items-start justify-between gap-3 px-6 py-4 border-b border-slate-100">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2.5 flex-wrap">
              <span className="font-mono text-[13px] font-bold text-teal-700 bg-teal-50 border border-teal-300 px-2.5 py-0.5 rounded-md tracking-wide">
                {survCode}
              </span>
              <h2 className="text-[15px] font-bold text-slate-900">{activeSurvey.name}</h2>
            </div>
            <p className="text-[12px] text-slate-600 font-mono">
              {[
                activeSurvey.water_body || activeSurvey.location || 'Coastal Zone',
                activeSurvey.vessel ? `Vessel: ${activeSurvey.vessel}` : null,
                activeSurvey.date   ? `Date: ${String(activeSurvey.date).slice(0, 10)}` : null,
              ].filter(Boolean).join(' · ')}
            </p>
          </div>
          <span className="font-mono text-[11.5px] bg-emerald-100 text-emerald-800 border border-emerald-300 px-3 py-1 rounded-full font-bold whitespace-nowrap self-start">
            {stats.total} {stats.total === 1 ? 'Detection' : 'Detections'} Logged
          </span>
        </div>

        {/* Statistics grid — colored cells, gapped, rounded */}
        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-3 px-6 py-5 bg-white">
          <StatCell value={stats.total}     label="TOTAL"     num="text-slate-800"   bg="bg-slate-100" />
          <StatCell value={stats.high}      label="HIGH"      num="text-rose-700"    bg="bg-rose-50" />
          <StatCell value={stats.medium}    label="MEDIUM"    num="text-amber-700"   bg="bg-amber-50" />
          <StatCell value={stats.low}       label="LOW"       num="text-emerald-700" bg="bg-emerald-50" />
          <StatCell value={stats.confirmed} label="CONFIRMED" num="text-teal-700"    bg="bg-teal-50" />
          <StatCell value={stats.cleaned}   label="CLEANED"   num="text-indigo-700"  bg="bg-indigo-50" />
          <StatCell value={stats.verified}  label="VERIFIED"  num="text-violet-700"  bg="bg-violet-50" />
        </div>

        {/* Export actions — teal / white / blue */}
        <div className="flex flex-wrap items-center gap-2.5 px-6 py-4 border-t border-slate-100">
          {/* Primary — PDF (teal, filled) */}
          <button
            onClick={() => handleExport('pdf')}
            className="group inline-flex items-center gap-2 h-9 px-4 bg-teal-600 hover:bg-teal-500 active:bg-teal-700 text-white text-[12px] font-semibold rounded-lg shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-150 border border-teal-700/40"
          >
            <DownloadIcon className="transition-transform group-hover:-translate-y-0.5" />
            Print / Save PDF
          </button>

          {/* Secondary — CSV (light blue) */}
          <button
            onClick={() => handleExport('csv')}
            className="group inline-flex items-center gap-2 h-9 px-4 border border-sky-200 bg-sky-50 hover:bg-sky-100 hover:border-sky-300 active:bg-sky-200 text-sky-700 text-[12px] font-semibold rounded-lg shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-150"
          >
            <SpreadsheetIcon />
            Export CSV
          </button>

          {/* Tertiary — JSON (solid blue) */}
          <button
            onClick={() => handleExport('json')}
            className="group inline-flex items-center gap-2 h-9 px-4 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-[12px] font-semibold rounded-lg shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-150 border border-blue-700/40"
          >
            <CodeIcon />
            Export JSON
          </button>
        </div>
      </div>

      {/* ═══ Detected Objects Card ═══════════════════════ */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">

        {/* Heading row */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <div>
            <h3 className="text-[14px] font-bold text-slate-900">Detected Objects</h3>
            <p className="text-[11.5px] text-slate-500 mt-0.5">
              Objects detected by the AI scanner for {survCode}
            </p>
          </div>
          <span className="text-[12px] font-mono font-semibold text-slate-600 bg-slate-100 border border-slate-200 px-2.5 py-1 rounded-lg">
            {scopedDetections.length} total
          </span>
        </div>

        {/* ── Filter toolbar ─────────────────────────── */}
        <div className="px-6 py-3.5 border-b border-slate-200 bg-slate-50/60">
          <div className="flex flex-wrap items-center gap-2.5">

            {/* Object Class */}
            <SelectWrapper>
              <select
                value={filterClass}
                onChange={(e) => setFilterClass(e.target.value)}
                className={`${SEL_CLS} pr-8 ${filterClass ? 'border-teal-300 bg-teal-50 text-teal-700' : ''}`}
              >
                <option value="">All Classes</option>
                {objectClasses.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </SelectWrapper>

            {/* Risk Level */}
            <SelectWrapper>
              <select
                value={filterRisk}
                onChange={(e) => setFilterRisk(e.target.value)}
                className={`${SEL_CLS} pr-8 ${filterRisk ? 'border-teal-300 bg-teal-50 text-teal-700' : ''}`}
              >
                <option value="">All Risk Levels</option>
                <option value="HIGH">🔴 High</option>
                <option value="MEDIUM">🟡 Medium</option>
                <option value="LOW">🟢 Low</option>
              </select>
            </SelectWrapper>

            {/* Detection Status */}
            <SelectWrapper>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className={`${SEL_CLS} pr-8 ${filterStatus ? 'border-teal-300 bg-teal-50 text-teal-700' : ''}`}
              >
                <option value="">All Statuses</option>
                <option value="pending">Pending</option>
                <option value="flagged">Flagged</option>
                <option value="confirmed">Confirmed</option>
                <option value="accepted">Accepted</option>
                <option value="verified">Verified</option>
                <option value="rejected">Rejected</option>
                <option value="cleaned">Cleaned</option>
              </select>
            </SelectWrapper>

            {/* Cleanup Verification */}
            <SelectWrapper>
              <select
                value={filterVerify}
                onChange={(e) => setFilterVerify(e.target.value)}
                className={`${SEL_CLS} pr-8 ${filterVerify ? 'border-violet-300 bg-violet-50 text-violet-700' : ''}`}
              >
                <option value="">Cleanup Verification</option>
                <option value="verified">Verified Only</option>
                <option value="unverified">Not Yet Verified</option>
              </select>
            </SelectWrapper>

            {/* Search */}
            <div className="relative">
              <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400 pointer-events-none"
                viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <input
                type="text"
                value={filterSearch}
                onChange={(e) => setFilterSearch(e.target.value)}
                placeholder="Search ID or class…"
                className={`${INPUT_CLS} pl-9 w-48 ${filterSearch ? 'border-teal-300' : ''}`}
              />
            </div>

            {/* Clear filters */}
            {hasActiveFilter && (
              <button
                onClick={() => { setFilterClass(''); setFilterRisk(''); setFilterStatus(''); setFilterVerify(''); setFilterSearch(''); }}
                className="inline-flex items-center gap-1 h-9 px-3 border border-rose-200 bg-rose-50 hover:bg-rose-100 text-rose-600 text-[11.5px] font-semibold rounded-lg transition-colors"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                </svg>
                Clear
              </button>
            )}

            {/* Showing count — pushed right */}
            <span className="ml-auto text-[12px] font-mono font-semibold text-slate-600 whitespace-nowrap bg-white border border-slate-200 px-2.5 py-1.5 rounded-lg">
              {filteredDetections.length} / {scopedDetections.length}
            </span>
          </div>
        </div>

        {/* ── Detection table ────────────────────────── */}
        {scopedDetections.length === 0 ? (
          <div className="py-16 text-center">
            <EmptyState icon="🔬" message="No detections recorded for this survey." />
          </div>
        ) : filteredDetections.length === 0 ? (
          <div className="py-12 text-center">
            <EmptyState icon="🔎" message="No detections match the current filters." />
          </div>
        ) : (
          <div className="table-scroll">
            <table className="w-full text-left text-[12px] border-collapse">
              <thead>
                <tr className="border-b-2 border-slate-200 bg-slate-100 text-[10.5px] font-mono font-bold text-slate-600 uppercase tracking-wider">
                  {/* ── Always visible (mobile + desktop) ── */}
                  <th className="py-3 px-4">Detection ID</th>
                  <th className="py-3 px-4">Object Class</th>
                  <th className="py-3 px-4">Risk Level</th>
                  <th className="py-3 px-4">Status</th>
                  {/* ── Desktop only (hidden on mobile) ── */}
                  <th className="py-3 px-4 hidden md:table-cell">Confidence</th>
                  <th className="py-3 px-4 hidden md:table-cell">Risk Score</th>
                  <th className="py-3 px-4 hidden lg:table-cell">Coordinates</th>
                  <th className="py-3 px-4 hidden lg:table-cell">Depth</th>
                  <th className="py-3 px-4 hidden md:table-cell">Cleanup Verification</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredDetections.map((det, i) => {
                  const rl     = riskLevel(det);
                  const sts    = String(det.status || 'pending').toLowerCase();
                  const vr     = verifyMap[String(det.id)];
                  const vBadge = verifyBadge(vr ? vr.result : 'Not Verified');
                  const verifyConf = vr ? `${((vr.confidence || 0) * 100).toFixed(0)}%` : null;

                  return (
                    <tr
                      key={det.id}
                      className="transition-colors hover:bg-teal-50/40 bg-white"
                    >
                      {/* ── Always visible ── */}

                      {/* Detection ID */}
                      <td className="py-3 px-4 font-mono font-bold text-teal-700 text-[11.5px] whitespace-nowrap">
                        {fmtDetCode(det)}
                      </td>

                      {/* Object class */}
                      <td className="py-3 px-4 font-semibold text-slate-800 capitalize whitespace-nowrap">
                        {det.object_class || det.objectClass || 'Unknown'}
                      </td>

                      {/* Risk level */}
                      <td className="py-3 px-4 whitespace-nowrap">
                        <span className={`font-mono text-[10.5px] font-bold px-2.5 py-0.5 rounded-full ${RISK_BADGE[rl]}`}>
                          {rl}
                        </span>
                      </td>

                      {/* Detection status */}
                      <td className="py-3 px-4 whitespace-nowrap">
                        <span className={`font-mono text-[10.5px] font-bold px-2.5 py-0.5 rounded-full uppercase ${statusBadgeCls(sts)}`}>
                          {sts}
                        </span>
                      </td>

                      {/* ── Desktop only ── */}

                      {/* Confidence */}
                      <td className="py-3 px-4 font-mono text-slate-700 whitespace-nowrap hidden md:table-cell">
                        {fmtConfidence(det.confidence)}
                      </td>

                      {/* Risk score */}
                      <td className="py-3 px-4 font-mono font-bold text-slate-800 whitespace-nowrap hidden md:table-cell">
                        {fmtRiskScore(det.risk_score ?? det.riskScore)}
                      </td>

                      {/* Coordinates */}
                      <td className="py-3 px-4 font-mono text-slate-600 text-[11px] whitespace-nowrap hidden lg:table-cell">
                        {fmtCoords(det.latitude ?? det.lat, det.longitude ?? det.lng)}
                      </td>

                      {/* Depth */}
                      <td className="py-3 px-4 font-mono text-slate-600 whitespace-nowrap hidden lg:table-cell">
                        {fmtDepth(det.depth ?? det.depthMeters)}
                      </td>

                      {/* Cleanup verification */}
                      <td className="py-3 px-4 whitespace-nowrap hidden md:table-cell">
                        <div className="flex flex-col gap-0.5">
                          <span className={`inline-flex items-center gap-1 font-mono text-[10.5px] font-bold px-2.5 py-0.5 rounded-full ${vBadge.cls}`}>
                            <span
                              className={`w-1.5 h-1.5 rounded-full ${
                                vr && ['cleared', 'potentially_removed'].includes(
                                  String(vr.result || '').toLowerCase().replace(/ /g, '_')
                                )
                                  ? 'bg-emerald-500'
                                  : 'bg-rose-500'
                              }`}
                            />
                            {vBadge.label}
                          </span>
                          {verifyConf && (
                            <span className="text-[10px] font-mono text-slate-500 pl-1">
                              conf. {verifyConf}
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
};

/* ─────────────────────────────────────────────────────────────
   Sub-components
───────────────────────────────────────────────────────────── */

/** Top toolbar: title + survey selector */
function ToolbarHeader({ surveys, selectedSurveyId, onSurveyChange }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b-2 border-slate-200">
      <div>
        <h1 className="text-[18px] font-bold text-slate-900 leading-tight">Survey Reports</h1>
        <p className="text-[12.5px] text-slate-500 mt-0.5">Generate official survey reports.</p>
      </div>
      <div className="flex items-center gap-2.5">
        <span className="text-[12.5px] text-slate-700 font-semibold whitespace-nowrap">Survey:</span>
        <SelectWrapper>
          <select
            value={selectedSurveyId}
            onChange={onSurveyChange}
            className={`${SEL_CLS} pr-8 min-w-[200px] font-mono font-semibold`}
          >
            {surveys.length === 0 && <option value="">No surveys</option>}
            {surveys.map((s) => (
              <option key={s.id} value={String(s.id)}>
                {fmtSurvCode(s.id)} · {s.name}
              </option>
            ))}
          </select>
        </SelectWrapper>
      </div>
    </div>
  );
}

/** Individual statistic cell — flat colored block, gapped, rounded corners, no icon */
function StatCell({ value, label, num, bg }) {
  return (
    <div className={`${bg} rounded-xl flex flex-col items-center justify-center py-4 px-2 transition-transform hover:-translate-y-0.5`}>
      <span className={`font-mono text-[23px] font-bold leading-none ${num}`}>{value}</span>
      <span className="text-[9.5px] font-mono font-semibold text-slate-500 mt-1.5 tracking-wider text-center">{label}</span>
    </div>
  );
}

/** Empty / placeholder message */
function EmptyState({ icon, message }) {
  return (
    <div className="flex flex-col items-center gap-2.5 py-2">
      <span className="text-4xl">{icon}</span>
      <p className="text-[13px] text-slate-500 font-medium">{message}</p>
    </div>
  );
}

/* Icon components */
function DownloadIcon({ className = '' }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" className={`w-3.5 h-3.5 ${className}`}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function SpreadsheetIcon({ className = '' }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`w-3.5 h-3.5 ${className}`}>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <line x1="3" y1="9" x2="21" y2="9" />
      <line x1="3" y1="15" x2="21" y2="15" />
      <line x1="9" y1="3" x2="9" y2="21" />
    </svg>
  );
}

function CodeIcon({ className = '' }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`w-3.5 h-3.5 ${className}`}>
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  );
}

export default ReportsPage;