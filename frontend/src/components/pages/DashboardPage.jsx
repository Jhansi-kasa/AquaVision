import React from 'react';
import { StatCard } from '../common/StatCard';
import { Badge } from '../common/Badge';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
export const DashboardPage = ({
  onNavigate,
  overviewData,
  surveys = [],
  currentSurveyId = null,
  onSelectSurvey,
}) => {
  const {
    activeSurvey = null,
    riskDistribution,
    recentDetections = [],
    objectDistribution = [],
    stats = {},
    loading = false,
    error = null,
  } = overviewData;

  const safeRisk = riskDistribution || {
    high: { count: 0, percent: 0 },
    medium: { count: 0, percent: 0 },
    low: { count: 0, percent: 0 },
  };

  const survey = activeSurvey || {
    id: '—',
    name: '—',
    vessel: '—',
    area: '—',
    date: '—',
    areaCoveredKm2: 0,
    avgConfidence: 0,
    cleanupProgress: { completed: 0, total: 0 },
  };

  const totalDetections = Number(stats.totalDetections ?? recentDetections.length);
  const highRisk = Number(stats.highRisk ?? safeRisk.high.count);
  const pendingReviews = Number(stats.pendingReviews ?? recentDetections.filter(d =>
    String(d.status || '').toLowerCase().includes('pending')
  ).length);
  const verifiedRemoved = Number(stats.verifiedRemoved ?? recentDetections.filter(d =>
    ['verified', 'cleaned', 'confirmed', 'removed'].includes(String(d.status || '').toLowerCase())
  ).length);

  return (
    <section className="animate-page-fade space-y-6 bg-[#EFF8FB] min-h-screen -m-6 p-6">
      {loading && (
        <div className="bg-white border border-slate-200 rounded-xl p-4 text-sm text-slate-800 font-semibold flex items-center gap-2.5 shadow-sm">
          <span className="w-4 h-4 border-2 border-teal-200 border-t-teal-600 rounded-full animate-spin flex-shrink-0" />
          Loading live telemetry and survey mission data from backend…
        </div>
      )}
      {error && !loading && (
        <div className="bg-rose-50 border border-rose-300 rounded-xl p-4 text-sm text-rose-800 font-semibold">
          Backend data could not be loaded: {error}
        </div>
      )}

      {/* STAT CARDS ROW */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <StatCard label="Total detections" value={totalDetections} delayIndex={0} />
        <StatCard label="High risk objects" value={highRisk} variant="risk" pad={2} delayIndex={1} />
        <StatCard label="Pending reviews" value={pendingReviews} variant="pending" pad={2} delayIndex={2} />
        <StatCard label="Verified removed" value={verifiedRemoved} variant="verified" delayIndex={3} />
      </div>

      {/* ACTIVE SURVEY & RISK DISTRIBUTION */}
      <div className="grid grid-cols-1 lg:grid-cols-[1.3fr_1fr] gap-3.5">
        {/* Active Survey Card */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 md:p-5 shadow-sm hover:shadow-md transition-shadow duration-300">

  {/* Header */}
  <div className="flex items-center justify-between pb-3 border-b border-slate-100">
    <div className="flex items-center gap-2">
      <span className="text-[13px] font-bold text-slate-900">
        Survey Analysis
      </span>
    </div>

    {/* Current Survey ID */}
    <span className="font-mono text-[11px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2.5 py-1 rounded-md">
      Active Survey : {survey.id}
    </span>
  </div>

  {/* Survey Selector */}
  {surveys.length > 1 && onSelectSurvey && (
    <div className="pt-3">
      <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1.5">
        Select Survey
      </label>

      <select
        value={survey.rawId || currentSurveyId || ''}
        onChange={(e) => onSelectSurvey(Number(e.target.value))}
        className="w-full h-9 px-3 text-[12px] font-semibold text-slate-800
                   bg-slate-50 border border-slate-200 rounded-lg
                   hover:border-teal-300 hover:bg-white
                   focus:outline-none focus:ring-2 focus:ring-teal-500/20
                   focus:border-teal-500
                   transition-all duration-200 cursor-pointer"
        title="Switch Active Survey"
      >
        {surveys.map((s) => {
          const displayLabel = s.name && !String(s.name).startsWith('Survey Mission ') ? s.name : s.water_body || '';
          return (
            <option key={s.id} value={s.id}>
              SURV-{String(s.id).padStart(3, '0')}{displayLabel ? ` · ${displayLabel}` : ''}
            </option>
          );
        })}
      </select>
    </div>
  )}
         <div className="space-y-2 text-[12.5px]">
            <br/><div className="flex justify-between py-2 border-b border-dashed border-slate-200">
              <span className="text-slate-700 font-semibold">Survey Name</span>
              <span className="font-mono font-bold text-slate-900">{survey.name && survey.name !== 'No active mission' && !String(survey.name).startsWith('Survey Mission ') ? survey.name : '—'}</span>
            </div>
            <div className="flex justify-between py-2 border-b border-dashed border-slate-200">
              <span className="text-slate-700 font-semibold">Vessel Name</span>
              <span className="font-mono font-bold text-slate-900">{survey.vessel || '—'}</span>
            </div>
            <div className="flex justify-between py-2 border-b border-dashed border-slate-200">
              <span className="text-slate-700 font-semibold">Survey area</span>
              <span className="font-mono font-bold text-slate-900">{survey.area && survey.area !== 'No active survey zone' && survey.area !== 'Coastal Survey Sector' ? survey.area : '—'}</span>
            </div>
            <div className="flex justify-between py-2 border-b border-dashed border-slate-200">
              <span className="text-slate-700 font-semibold">Survey date</span>
              <span className="font-mono font-bold text-slate-900">{survey.date}</span>
            </div>
            <div className="flex justify-between py-2 border-b border-dashed border-slate-200">
              <span className="text-slate-700 font-semibold">Area covered</span>
              <span className="font-mono font-bold text-slate-900">{survey.areaCoveredKm2} km²</span>
            </div>
            <div className="flex justify-between py-2 border-b border-dashed border-slate-200">
              <span className="text-slate-700 font-semibold">Avg. AI confidence</span>
              <span className="font-mono font-bold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
                {survey.avgConfidence}%
              </span>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-slate-700 font-semibold">Cleanup progress</span>
              <span className="font-mono font-bold text-slate-900">
                {survey.cleanupProgress?.completed ?? 0} / {survey.cleanupProgress?.total ?? 0}
              </span>
            </div>
          </div>

          <div className="h-2 bg-slate-100 border border-slate-200 rounded-full overflow-hidden mt-2.5">
            <div
              className="h-full bg-gradient-to-r from-teal-500 to-teal-400 rounded-full transition-all duration-500"
              style={{
                width: `${survey.cleanupProgress?.total ? (survey.cleanupProgress.completed / survey.cleanupProgress.total) * 100 : 0}%`,
              }}
            />
          </div>
        </div>
      
       {/* Risk Distribution */}
<div className="bg-white border border-slate-200 rounded-xl p-4 md:p-5 shadow-sm hover:shadow-md transition-shadow duration-300">

  {/* Header */}
  <div className="flex items-center justify-between pb-3 border-b border-slate-100">
    <span className="text-[13px] font-bold text-slate-900">
      Risk Distribution
    </span>

    <span className="text-[11px] font-mono text-slate-600 font-bold">
      Classified Targets
    </span>
  </div>

  {/* Large Pie */}
  <div className="flex justify-center pt-4">
    <div className="w-full max-w-[270px] h-[270px]">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={[
              { name: 'High', value: safeRisk.high.count },
              { name: 'Medium', value: safeRisk.medium.count },
              { name: 'Low', value: safeRisk.low.count },
            ]}
            cx="50%"
            cy="50%"
            outerRadius={120}
            paddingAngle={2}
            dataKey="value"
            stroke="#ffffff"
            strokeWidth={2}
          >
            <Cell fill="#F43F5E" />
            <Cell fill="#F59E0B" />
            <Cell fill="#10B981" />
          </Pie>

          <Tooltip
            contentStyle={{
              borderRadius: '8px',
              border: '1px solid #E2E8F0',
              fontSize: '12px',
              fontWeight: 600,
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  </div>

  {/* Values */}
  <div className="grid grid-cols-3 gap-3 mt-2 pt-3 border-t border-slate-100">

    {/* High */}
    <div className="text-center">
      <div className="flex items-center justify-center gap-1.5 mb-1">
        <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
        <span className="text-[12px] font-bold text-slate-700">
          High
        </span>
      </div>

      <div className="text-[18px] font-mono font-bold text-slate-900">
        {safeRisk.high.count}
      </div>

      <div className="text-[10px] font-semibold text-slate-500">
        {safeRisk.high.percent}%
      </div>
    </div>

    {/* Medium */}
    <div className="text-center">
      <div className="flex items-center justify-center gap-1.5 mb-1">
        <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
        <span className="text-[12px] font-bold text-slate-700">
          Medium
        </span>
      </div>

      <div className="text-[18px] font-mono font-bold text-slate-900">
        {safeRisk.medium.count}
      </div>

      <div className="text-[10px] font-semibold text-slate-500">
        {safeRisk.medium.percent}%
      </div>
    </div>

    {/* Low */}
    <div className="text-center">
      <div className="flex items-center justify-center gap-1.5 mb-1">
        <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
        <span className="text-[12px] font-bold text-slate-700">
          Low
        </span>
      </div>

      <div className="text-[18px] font-mono font-bold text-slate-900">
        {safeRisk.low.count}
      </div>

      <div className="text-[10px] font-semibold text-slate-500">
        {safeRisk.low.percent}%
      </div>
    </div>

  </div>
</div>
</div>
      {/* RECENT DETECTIONS & OBJECT DISTRIBUTION */}
      <div className="grid grid-cols-1 lg:grid-cols-[1.3fr_1fr] gap-3.5">
        {/* Recent Detections Table */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 md:p-5 shadow-sm hover:shadow-md transition-shadow duration-300">
          <div className="text-[13px] font-bold text-slate-900 mb-3.5 flex justify-between items-center pb-2 border-b border-slate-100">
            <span>Recent detections</span>
            <span className="font-mono text-[11px] text-slate-600 font-bold bg-slate-100 px-2 py-0.5 rounded">
              latest 5
            </span>
          </div>
          <div className="table-scroll">
          <table className="w-full text-left text-[12.5px] border-collapse min-w-[420px]">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/70">
                <th className="p-[9px_12px] text-[11px] text-slate-800 font-bold">ID</th>
                <th className="p-[9px_12px] text-[11px] text-slate-800 font-bold">Object</th>
                <th className="p-[9px_12px] text-[11px] text-slate-800 font-bold">Confidence</th>
                <th className="p-[9px_12px] text-[11px] text-slate-800 font-bold">Risk</th>
                <th className="p-[9px_12px] text-[11px] text-slate-800 font-bold">Status</th>
              </tr>
            </thead>
            <tbody>
              {recentDetections.length ? (
                recentDetections.map((det) => (
                  <tr
                    key={det.id}
                    onClick={() => onNavigate('review')}
                    className="border-b border-slate-100 last:border-b-0 hover:bg-teal-50/50 cursor-pointer transition-colors"
                  >
                    <td className="p-[11px_12px] font-mono font-bold text-slate-900">{det.id}</td>
                    <td className="p-[11px_12px] font-semibold text-slate-800">{det.objectClass}</td>
                    <td className="p-[11px_12px] font-mono font-bold text-slate-900">{det.confidence}%</td>
                    <td className="p-[11px_12px]"><Badge level={det.risk} /></td>
                    <td className="p-[11px_12px]">
                      <span className={`status-pill ${String(det.status).toLowerCase() === 'reviewed' ? 'reviewed' : ''}`}>
                        {det.status}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" className="p-6 text-center text-slate-600 font-semibold">
                    No detections returned by the backend.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          </div>
        </div>

        {/* Object Distribution */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 md:p-5 shadow-sm hover:shadow-md transition-shadow duration-300">
          <div className="text-[13px] font-bold text-slate-900 mb-3.5 pb-2 border-b border-slate-100 flex justify-between items-center">
            <span>Object distribution</span>
            <span className="text-[11px] font-mono text-slate-600 font-bold">Acoustic Classes</span>
          </div>
          <div className="space-y-1 text-[12.5px]">
            {objectDistribution.length ? (
              objectDistribution.map((obj) => (
                <div
                  key={obj.name}
                  className="flex justify-between items-center py-2 border-b border-slate-100 last:border-b-0 hover:bg-slate-50 px-2 rounded transition-colors"
                >
                  <span className="text-slate-900 font-bold">{obj.name}</span>
                  <span className="font-mono text-teal-800 font-bold bg-teal-50 border border-teal-200 px-2 py-0.5 rounded text-xs">
                    {obj.count}
                  </span>
                </div>
              ))
            ) : (
              <div className="py-4 text-slate-600 font-semibold text-center">No object data available.</div>
            )}
          </div>
        </div>
      </div>

      {/* QUICK ACTIONS ROW */}
      <div>
        <h2 className="text-[15px] font-bold text-slate-900 mb-3 flex items-center gap-2">
          <span>Quick actions</span>
          <span className="text-[11px] font-normal text-slate-600">(Core Mission Workflows)</span>
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {[
            {
              page: 'surveys',
              label: 'Start new survey',
              subtitle: 'Initialize mission',
              bgClass: 'bg-teal-50 text-teal-700 border-teal-200 group-hover:bg-teal-600 group-hover:text-white',
              icon: (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="w-8 h-8">
                  <path d="M12 4v16m-8-8h16" />
                  <circle cx="12" cy="12" r="9" />
                </svg>
              ),
            },
            {
              page: 'sonar',
              label: 'Upload sonar',
              subtitle: 'YOLO acoustic scan',
              bgClass: 'bg-cyan-50 text-cyan-700 border-cyan-200 group-hover:bg-cyan-600 group-hover:text-white',
              icon: (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="w-8 h-8">
                  <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                  <path d="M12 12v9" />
                  <path d="m16 16-4-4-4 4" />
                </svg>
              ),
            },
            {
              page: 'map',
              label: 'View risk map',
              subtitle: 'GIS spatial telemetry',
              bgClass: 'bg-emerald-50 text-emerald-700 border-emerald-200 group-hover:bg-emerald-600 group-hover:text-white',
              icon: (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="w-8 h-8">
                  <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6" />
                  <line x1="8" y1="2" x2="8" y2="18" />
                  <line x1="16" y1="6" x2="16" y2="22" />
                </svg>
              ),
            },
            {
              page: 'review',
              label: 'Review detections',
              subtitle: 'Human verification',
              bgClass: 'bg-amber-50 text-amber-700 border-amber-200 group-hover:bg-amber-600 group-hover:text-white',
              icon: (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="w-8 h-8">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="22" y1="12" x2="18" y2="12" />
                  <line x1="6" y1="12" x2="2" y2="12" />
                  <line x1="12" y1="6" x2="12" y2="2" />
                  <line x1="12" y1="22" x2="12" y2="18" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
              ),
            },
            {
              page: 'reports',
              label: 'Generate report',
              subtitle: 'Export PDF / CSV',
              bgClass: 'bg-indigo-50 text-indigo-700 border-indigo-200 group-hover:bg-indigo-600 group-hover:text-white',
              icon: (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="w-8 h-8">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
              ),
            },
          ].map((item, i) => (
            <div
              key={item.page}
              onClick={() => onNavigate(item.page)}
              className={`group bg-white border border-slate-200 rounded-xl p-4 text-center cursor-pointer transition-all duration-200 hover:border-teal-500 hover:shadow-lg hover:-translate-y-1 flex flex-col items-center justify-center min-h-[120px] ${
                i === 4 ? 'col-span-2 sm:col-span-1' : ''
              }`}
            >
              <div
                className={`w-14 h-14 rounded-2xl border flex items-center justify-center mb-2.5 transition-all duration-200 shadow-sm ${item.bgClass}`}
              >
                {item.icon}
              </div>
              <div className="text-[13px] font-bold text-slate-900 group-hover:text-teal-700 leading-tight">
                {item.label}
              </div>
              <div className="text-[10.5px] text-slate-600 font-semibold mt-1">
                {item.subtitle}
              </div>
            </div>
          ))}
        </div>
      </div>
        </section>
  );
};
