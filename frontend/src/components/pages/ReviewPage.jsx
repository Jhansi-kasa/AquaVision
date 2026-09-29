import React, { useMemo, useState, useEffect } from 'react';
import { useToast } from '../../context/ToastContext';
import { api } from '../../services/api';
import {
  normalizeDetection,
  getConfidenceCategory,
  getRiskBadgeStyle,
  formatConfidence,
  formatCoordinate,
} from '../../utils/detectionUtils';

const CONFIRMED_STATUSES = ['confirmed', 'verified', 'accepted'];

export const ReviewPage = ({
  analysis,
  allDetectionsList = [],
  initialSelectedDetectionId = null,
  onClearSelection,
  currentSurveyId,
  onReviewed,
  onNavigate,
}) => {
  const { showToast } = useToast();

  // Fetched sonar image for the currently selected detection (used when navigating from list)
  const [detectionImageSrc, setDetectionImageSrc] = useState(null);

  // 11. When arriving from "Open review" in Sonar Analysis, automatically open/select that detection
  const [selectedDetectionId, setSelectedDetectionId] = useState(initialSelectedDetectionId);

  // 8 & 10. Default filter must be 'pending' whenever opened normally
  const [filterStatus, setFilterStatus] = useState('pending'); // 'all' | 'pending' | 'confirmed' | 'rejected' | 'flagged'
  const [searchQuery, setSearchQuery] = useState('');
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [activeBanner, setActiveBanner] = useState(null);

  // Local detection state for real-time reactivity
  const [localDetections, setLocalDetections] = useState([]);

  // Sync initialSelectedDetectionId prop if changed from parent
  useEffect(() => {
    if (initialSelectedDetectionId) {
      setSelectedDetectionId(initialSelectedDetectionId);
    }
  }, [initialSelectedDetectionId]);

  // Fetch sonar image from backend when a detection is selected
  useEffect(() => {
    const analysisSrc =
      analysis?.detectionImage ||
      analysis?.detection_image ||
      analysis?.processedImage ||
      analysis?.processed_image ||
      analysis?.uploadedImage ||
      null;

    const detectionInAnalysis =
      analysis?.detections &&
      analysis.detections.some((d) => String(d.id) === String(selectedDetectionId));

    if (detectionInAnalysis && analysisSrc) {
      // Current selected detection is directly from the fresh analysis scan
      setDetectionImageSrc(null);
      return;
    }

    if (!selectedDetectionId) {
      setDetectionImageSrc(null);
      return;
    }

    const currentDet = localDetections.find((d) => String(d.id) === String(selectedDetectionId));
    let cancelled = false;
    api
      .getDetectionImage(currentDet || selectedDetectionId, currentDet?.image_id)
      .then((result) => {
        if (!cancelled) setDetectionImageSrc(result?.data_url || null);
      })
      .catch(() => {
        if (!cancelled) setDetectionImageSrc(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedDetectionId, analysis, localDetections]);

  // Synchronize detections from allDetectionsList (full set) or analysis (fallback)
  // IMPORTANT: Always prefer allDetectionsList — it is the complete merged set maintained by App.jsx.
  // analysis.detections only contains detections from the most recent single scan, so using it
  // as the primary source causes "Back to Pending List" to show only that scan's detections.
  useEffect(() => {
    const surveyMeta = analysis?.survey || {
      survey_id: analysis?.survey_id || currentSurveyId,
      latitude: analysis?.survey?.latitude,
      longitude: analysis?.survey?.longitude,
      depth: analysis?.survey?.depth,
    };

    let rawList = [];
    if (allDetectionsList && Array.isArray(allDetectionsList) && allDetectionsList.length > 0) {
      // Primary: full list maintained by App.jsx (includes all scans)
      rawList = allDetectionsList;
    } else if (analysis?.detections && Array.isArray(analysis.detections) && analysis.detections.length > 0) {
      // Fallback: current analysis scan only (e.g. first scan before rawDetections is populated)
      rawList = analysis.detections;
    }

    const normalizedList = rawList.map((d) => normalizeDetection(d, surveyMeta)).filter(Boolean);

    setLocalDetections(normalizedList);
  }, [analysis, allDetectionsList, currentSurveyId]);

  // Counts for tabs
  const counts = useMemo(() => {
    const total = localDetections.length;
    const pending = localDetections.filter((d) => d.status === 'pending').length;
    const confirmed = localDetections.filter((d) => CONFIRMED_STATUSES.includes(d.status)).length;
    const rejected = localDetections.filter((d) => d.status === 'rejected').length;
    const flagged = localDetections.filter((d) => d.status === 'flagged').length;
    return { total, pending, confirmed, rejected, flagged };
  }, [localDetections]);

  // 8, 9, 10. Filtered list for table (Defaulting to Pending)
  const displayedDetections = useMemo(() => {
    return localDetections.filter((d) => {
      // Status filter
      if (filterStatus === 'pending' && d.status !== 'pending') return false;
      if (filterStatus === 'confirmed' && !CONFIRMED_STATUSES.includes(d.status)) return false;
      if (filterStatus === 'rejected' && d.status !== 'rejected') return false;
      if (filterStatus === 'flagged' && d.status !== 'flagged') return false;

      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesId = String(d.detection_id || '').toLowerCase().includes(q);
        const matchesClass = String(d.object_class || '').toLowerCase().includes(q);
        return matchesId || matchesClass;
      }
      return true;
    });
  }, [localDetections, filterStatus, searchQuery]);

  // Active detection for detail view
  const currentDetection = useMemo(() => {
    if (!selectedDetectionId) return null;
    return localDetections.find((d) => String(d.id) === String(selectedDetectionId)) || null;
  }, [localDetections, selectedDetectionId]);

  const currentIndex = useMemo(() => {
    if (!currentDetection) return -1;
    return localDetections.findIndex((d) => String(d.id) === String(currentDetection.id));
  }, [localDetections, currentDetection]);

  // Detections that have already been reviewed/confirmed must never be re-submitted.
  const isCurrentAlreadyConfirmed = !!currentDetection && CONFIRMED_STATUSES.includes(currentDetection.status);

  // 13, 14, 15, 16. Actions: Accept, Reject, Flag
  const handleAction = async (verdict) => {
    if (!currentDetection) return;

    // Guard: a detection that is already confirmed/verified/accepted cannot be reviewed again.
    if (CONFIRMED_STATUSES.includes(currentDetection.status)) {
      showToast({
        type: 'warning',
        message: `${currentDetection.full_identifier || currentDetection.detection_id} has already been reviewed and confirmed. No further action allowed.`,
      });
      return;
    }

    if (submitting) return;

    setSubmitting(true);
    try {
      const response = await api.submitDetectionReview(currentDetection.id, {
        verdict,
        reviewer: 'sonar-operator',
        comment:
          comment ||
          (verdict === 'confirmed'
            ? 'Accepted detection — confirmed for cleanup'
            : verdict === 'flagged'
            ? 'Flagged for inspection pass'
            : 'Rejected false acoustic reflection'),
      });

      const newStatus =
        response?.new_status ||
        response?.status ||
        (verdict === 'confirmed' ? 'confirmed' : verdict === 'flagged' ? 'flagged' : 'rejected');

      // Update local state
      setLocalDetections((prev) =>
        prev.map((item) =>
          String(item.id) === String(currentDetection.id) ? { ...item, status: newStatus } : item
        )
      );

      setActiveBanner(verdict);
      onReviewed?.(currentDetection.id, newStatus);

      showToast({
        type: verdict === 'confirmed' ? 'success' : verdict === 'flagged' ? 'info' : 'warning',
        message:
          verdict === 'confirmed'
            ? `${currentDetection.full_identifier || currentDetection.detection_id} ACCEPTED (Status: CONFIRMED). Available on Risk Map & Cleanup Priority.`
            : verdict === 'flagged'
            ? `${currentDetection.full_identifier || currentDetection.detection_id} FLAGGED for inspection.`
            : `${currentDetection.full_identifier || currentDetection.detection_id} REJECTED as false detection.`,
      });
    } catch (err) {
      showToast({ type: 'error', message: err?.message || 'Could not record review action.' });
    } finally {
      setSubmitting(false);
    }
  };

  const navigateDetection = (direction) => {
    if (currentIndex === -1) return;
    const nextIdx = currentIndex + direction;
    if (nextIdx >= 0 && nextIdx < localDetections.length) {
      setSelectedDetectionId(localDetections[nextIdx].id);
      setActiveBanner(null);
      setComment('');
    }
  };

  const handleCloseDetail = () => {
    setSelectedDetectionId(null);
    onClearSelection?.();
    setActiveBanner(null);
    setComment('');
  };

  // If no detections available anywhere
  if (localDetections.length === 0) {
    return (
      <section className="animate-page-fade space-y-4 max-w-[1360px] mx-auto">
        <div className="border-b border-slate-100 pb-2">
          <h1 className="text-[18px] font-bold text-slate-800">Human Verification</h1>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-8 text-center max-w-md mx-auto my-6">
          <div className="w-10 h-10 rounded-full bg-teal-50 text-teal-600 mx-auto flex items-center justify-center mb-2.5">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <h2 className="text-[14px] font-bold text-slate-800 mb-1">No Detections in Verification Queue</h2>
          <p className="text-[12px] text-slate-500 mb-4">
            Run a sonar scan on the Sonar Analysis page. Detections returned by the model will appear here for human confirmation.
          </p>
          <button onClick={() => onNavigate('sonar')} className="btn primary mx-auto text-[12px] py-1.5 px-3">
            Go to Sonar Analysis
          </button>
        </div>
      </section>
    );
  }

  // =========================================================================
  // VIEW MODE 1: COMPLETE DETAIL VIEW (When arriving from "Open review" or clicking a row)
  // =========================================================================
  if (currentDetection) {
    const confCat = getConfidenceCategory(currentDetection.confidence);
    const riskInfo = getRiskBadgeStyle(currentDetection.risk_level);
    const surveyId =
      currentDetection.survey_id ?? currentDetection.surveyId ?? analysis?.survey_id ?? currentSurveyId ?? '—';

    const isFromAnalysis =
      analysis?.detections &&
      analysis.detections.some((d) => String(d.id) === String(currentDetection.id));

    const frameSrc =
      detectionImageSrc ||
      (isFromAnalysis
        ? analysis?.detectionImage ||
          analysis?.detection_image ||
          analysis?.processedImage ||
          analysis?.processed_image ||
          analysis?.uploadedImage
        : null) ||
      analysis?.detectionImage ||
      analysis?.processedImage ||
      null;

    return (
      <section className="animate-page-fade space-y-6 bg-[#EFF8FB] min-h-screen -m-6 p-6">
        {/* Navigation Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2 pb-1 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <button
              onClick={handleCloseDetail}
              className="btn secondary flex items-center gap-1.5 text-[11.5px] py-1 px-2.5 font-medium"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" className="w-3.5 h-3.5">
                <line x1="19" y1="12" x2="5" y2="12" />
                <polyline points="12 19 5 12 12 5" />
              </svg>
              Back to Pending List
            </button>
            <span className="font-mono text-[11px] text-slate-500">
              Target {currentIndex + 1} of {localDetections.length}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => navigateDetection(-1)}
              disabled={currentIndex <= 0}
              className="btn py-1 px-2.5 text-[11.5px] disabled:opacity-40"
            >
              ← Prev
            </button>
            <button
              onClick={() => navigateDetection(1)}
              disabled={currentIndex >= localDetections.length - 1}
              className="btn py-1 px-2.5 text-[11.5px] disabled:opacity-40"
            >
              Next →
            </button>
          </div>
        </div>

        {/* 12. Complete Review Detail:
            Actual sonar image, thick green bounding box, detection ID, object class, AI confidence,
            confidence category, risk score, risk level, latitude, longitude, depth, current review status */}
        <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-4">
          {/* Left: Actual Sonar Image with THICK GREEN Bounding Box */}
          <div className="bg-[#030d12] border border-slate-800 rounded-xl p-3 relative shadow-md flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between font-mono text-[9.5px] text-slate-400 mb-2 px-1">
                <span className="text-[#00ff66] font-bold">
                  {currentDetection.full_identifier || currentDetection.detection_id} · ACOUSTIC TARGET FRAME
                </span>
                <span>SURVEY {surveyId}</span>
              </div>

              {/* Sonar Frame */}
              <div className="relative w-full aspect-[480/300] rounded overflow-hidden bg-[#02090d] flex items-center justify-center">
                <div
                  className="absolute inset-0 opacity-20 pointer-events-none"
                  style={{
                    backgroundImage:
                      'repeating-linear-gradient(100deg, rgba(45,212,196,.15) 0px, transparent 3px, transparent 8px)',
                  }}
                />

                {frameSrc ? (
                  <div className="relative w-full h-full flex items-center justify-center">
                    <img src={frameSrc} alt="Actual Sonar Detection" className="w-full h-full object-contain" />

                  </div>
                ) : (
                  <div className="text-center p-4 text-slate-500 font-mono text-xs">ACOUSTIC FRAME DATA</div>
                )}
              </div>
            </div>

            <div className="text-[10px] text-slate-400 font-mono mt-2 px-1 flex items-center justify-between">
              <span className="text-[#00ff66] font-bold">● High-visibility green bounding box</span>
              <span>Coordinates verified from survey</span>
            </div>
          </div>

          {/* Right: Detailed Inspection & 13. Accept, Reject, Flag */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col justify-between space-y-3">
            <div>
              {/* Header: Detection ID & Status */}
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-200">
                <div>
                  <span className="text-[10.5px] font-mono text-slate-700 font-bold uppercase">Target Identifier</span>
                  <h2 className="text-[17px] font-bold text-slate-900 leading-tight">
                    {currentDetection.full_identifier || currentDetection.detection_id}
                  </h2>
                </div>
                <span
                  className={`font-mono text-[10.5px] font-bold px-2 py-0.5 rounded-full border uppercase ${
                    currentDetection.status === 'confirmed'
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : currentDetection.status === 'rejected'
                      ? 'bg-rose-50 text-rose-700 border-rose-200'
                      : currentDetection.status === 'flagged'
                      ? 'bg-amber-50 text-amber-700 border-amber-200'
                      : 'bg-slate-100 text-slate-700 border-slate-300'
                  }`}
                >
                  {currentDetection.status}
                </span>
              </div>

              {/* 12. Complete attributes (Object class, AI confidence, Category, Risk score, Risk level, Lat, Lng, Depth, Status) */}
              <div className="space-y-1.5 text-[11.5px]">
                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-700 font-semibold">Object Class</span>
                  <span className="font-bold text-slate-900">{currentDetection.object_class}</span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-700 font-semibold">AI Confidence</span>
                  <span className="font-mono font-bold text-slate-900">{formatConfidence(currentDetection.confidence)}</span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100 items-center">
                  <span className="text-slate-700 font-semibold">Confidence Category</span>
                  <span className={`px-1.5 py-0.2 rounded font-mono text-[10px] font-bold border ${confCat.badgeClass}`}>
                    {confCat.category}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100 items-center">
                  <span className="text-slate-700 font-semibold">Risk Level</span>
                  <span className={`px-1.5 py-0.2 rounded font-mono text-[10px] font-bold border ${riskInfo.badgeClass}`}>
                    {currentDetection.risk_level}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-700 font-semibold">Risk Score</span>
                  <span className="font-mono font-bold text-slate-900">{currentDetection.risk_score} / 100</span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-700 font-semibold">Latitude</span>
                  <span className="font-mono font-semibold text-slate-900">{formatCoordinate(currentDetection.latitude)}</span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-700 font-semibold">Longitude</span>
                  <span className="font-mono font-semibold text-slate-900">{formatCoordinate(currentDetection.longitude)}</span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-700 font-semibold">Depth</span>
                  <span className="font-mono font-bold text-slate-900">
                    {currentDetection.depth != null ? `${currentDetection.depth} m` : '—'}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-700 font-semibold">Review Status</span>
                  <span className="font-mono font-bold text-teal-800 uppercase">{currentDetection.status}</span>
                </div>
              </div>

              {/* Reviewer Note */}
              <div className="mt-3">
                <label className="block text-[10.5px] font-mono text-slate-700 font-bold mb-0.5">Reviewer Comment</label>
                <textarea
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="Optional review notes..."
                  disabled={isCurrentAlreadyConfirmed}
                  className="w-full border border-slate-300 rounded p-1.5 text-[11px] font-medium text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-500 resize-none h-12 disabled:bg-slate-50 disabled:text-slate-400"
                />
              </div>

              {/* Already-reviewed notice — locks the detection from being reviewed again */}
              {isCurrentAlreadyConfirmed && (
                <div className="mt-2 p-1.5 bg-emerald-50 border border-emerald-200 rounded text-emerald-800 text-[10.5px] font-mono">
                  ✓ Already reviewed and confirmed. No further action needed.
                </div>
              )}

              {/* Feedback banner */}
              {activeBanner === 'confirmed' && (
                <div className="mt-2 p-1.5 bg-emerald-50 border border-emerald-200 rounded text-emerald-800 text-[10.5px] font-mono">
                  ✓ Confirmed! Made available to Risk Map &amp; Cleanup Priority.
                </div>
              )}
              {activeBanner === 'rejected' && (
                <div className="mt-2 p-1.5 bg-rose-50 border border-rose-200 rounded text-rose-800 text-[10.5px] font-mono">
                  ✕ Rejected. Excluded from Risk Map &amp; Cleanup Priority.
                </div>
              )}
              {activeBanner === 'flagged' && (
                <div className="mt-2 p-1.5 bg-amber-50 border border-amber-200 rounded text-amber-800 text-[10.5px] font-mono">
                  ⚑ Flagged for second pass. Available under Flagged filter.
                </div>
              )}
            </div>

            {/* 13. Provide: ✓ Accept, ✕ Reject, ⚑ Flag — locked once the detection is already confirmed */}
            <div className="pt-2 border-t border-slate-100">
              <div className="grid grid-cols-3 gap-2">
                {/* 14. ACCEPT */}
                <button
                  type="button"
                  onClick={() => handleAction('confirmed')}
                  disabled={submitting || isCurrentAlreadyConfirmed}
                  title={isCurrentAlreadyConfirmed ? 'Already reviewed — cannot confirm again' : undefined}
                  className="btn confirm bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-1.5 text-[11.5px] flex items-center justify-center gap-1 shadow-sm disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-emerald-600"
                >
                  ✓ Accept
                </button>

                {/* 15. REJECT */}
                <button
                  type="button"
                  onClick={() => handleAction('rejected')}
                  disabled={submitting || isCurrentAlreadyConfirmed}
                  title={isCurrentAlreadyConfirmed ? 'Already reviewed — no further action allowed' : undefined}
                  className="btn reject bg-rose-600 hover:bg-rose-700 text-white font-bold py-1.5 text-[11.5px] flex items-center justify-center gap-1 shadow-sm disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-rose-600"
                >
                  ✕ Reject
                </button>

                {/* 16. FLAG */}
                <button
                  type="button"
                  onClick={() => handleAction('flagged')}
                  disabled={submitting || isCurrentAlreadyConfirmed}
                  title={isCurrentAlreadyConfirmed ? 'Already reviewed — no further action allowed' : undefined}
                  className="btn investigate bg-amber-500 hover:bg-amber-600 text-white font-bold py-1.5 text-[11.5px] flex items-center justify-center gap-1 shadow-sm disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-amber-500"
                >
                  ⚑ Flag
                </button>
              </div>

              {currentIndex < localDetections.length - 1 && (
                <button
                  type="button"
                  onClick={() => navigateDetection(1)}
                  className="w-full text-center text-[10.5px] font-mono text-teal-700 hover:text-teal-800 pt-1.5"
                >
                  Next detection →
                </button>
              )}
            </div>
          </div>
        </div>
      </section>
    );
  }

  // =========================================================================
  // VIEW MODE 2: GENERIC TABLE VIEW (Defaults to Pending filter, Req 8 & 10)
  // =========================================================================
  return (
    <section className="animate-page-fade space-y-6 bg-[#EFF8FB] min-h-screen -m-6 p-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 pb-3 border-b border-slate-200">
        <p className="text-[15px] text-slate-900 font-medium">
          Verify AI-detected targets.
        </p>

        <div className="flex items-center gap-2 flex-shrink-0">
          <button onClick={() => onNavigate('sonar')} className="btn secondary text-[11.5px] py-1 px-2.5">
            ← Sonar Analysis
          </button>
          <button onClick={() => onNavigate('map')} className="btn primary text-[11.5px] py-1 px-2.5">
            Risk Map ({counts.confirmed} confirmed) →
          </button>
        </div>
      </div>

      {/* 9. Provide filters/tabs: All | Pending | Confirmed | Rejected | Flagged */}
      <div className="bg-white border border-slate-200 rounded-xl p-2.5 shadow-sm flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
          {/* ALL */}
          <button
            onClick={() => setFilterStatus('all')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              filterStatus === 'all' ? 'bg-teal-600 text-white font-bold shadow-sm' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            All ({counts.total})
          </button>

          {/* PENDING (Default per Req 8 & 10) */}
          <button
            onClick={() => setFilterStatus('pending')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              filterStatus === 'pending' ? 'bg-teal-600 text-white font-bold shadow-sm' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            Pending ({counts.pending})
          </button>

          {/* CONFIRMED */}
          <button
            onClick={() => setFilterStatus('confirmed')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              filterStatus === 'confirmed' ? 'bg-teal-600 text-white font-bold shadow-sm' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            Confirmed ({counts.confirmed})
          </button>

          {/* REJECTED */}
          <button
            onClick={() => setFilterStatus('rejected')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              filterStatus === 'rejected' ? 'bg-teal-600 text-white font-bold shadow-sm' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            Rejected ({counts.rejected})
          </button>

          {/* FLAGGED */}
          <button
            onClick={() => setFilterStatus('flagged')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              filterStatus === 'flagged' ? 'bg-teal-600 text-white font-bold shadow-sm' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            Flagged ({counts.flagged})
          </button>
        </div>

        {/* Search */}
        <div className="relative w-full sm:w-auto sm:min-w-[180px]">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter target..."
            className="w-full border border-slate-200 rounded-lg pl-7 pr-2.5 py-0.5 text-[11px] focus:outline-none focus:ring-1 focus:ring-teal-400 font-ui"
          />
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="w-3 h-3 text-slate-400 absolute left-2 top-1.5"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <div className="table-scroll">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-100 border-b border-slate-300 text-[10.5px] font-mono text-slate-800 uppercase tracking-wider font-bold">
                {/* ── Always visible (mobile + desktop) ── */}
                <th className="py-2.5 px-3">Detection ID</th>
                <th className="py-2.5 px-3">Object Class</th>
                {/* Confidence is now visible on mobile too, per request */}
                <th className="py-2.5 px-3">Confidence</th>
                {/* ── Desktop only ── */}
                <th className="py-2.5 px-3 hidden lg:table-cell">Category</th>
                <th className="py-2.5 px-3 hidden md:table-cell">Risk Level</th>
                <th className="py-2.5 px-3 hidden lg:table-cell">Risk Score</th>
                <th className="py-2.5 px-3 hidden lg:table-cell">Latitude</th>
                <th className="py-2.5 px-3 hidden lg:table-cell">Longitude</th>
                <th className="py-2.5 px-3 hidden lg:table-cell">Depth</th>
                <th className="py-2.5 px-3 hidden md:table-cell">Status</th>
                {/* ── Action: hidden entirely on the Confirmed tab — nothing left to action ── */}
                {filterStatus !== 'confirmed' && <th className="py-2.5 px-3 text-right">Action</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-[11.5px]">
              {displayedDetections.length === 0 ? (
                <tr>
                  <td
                    colSpan={filterStatus !== 'confirmed' ? 11 : 10}
                    className="py-6 text-center text-slate-600 font-mono text-[11px] font-semibold"
                  >
                    No detections found in "{filterStatus.toUpperCase()}" queue.
                  </td>
                </tr>
              ) : (
                displayedDetections.map((detection) => {
                  const confCat = getConfidenceCategory(detection.confidence);
                  const riskInfo = getRiskBadgeStyle(detection.risk_level);
                  // A confirmed detection has already been reviewed — no further review allowed
                  const isConfirmedDet = CONFIRMED_STATUSES.includes(String(detection.status || '').toLowerCase());

                  return (
                    <tr
                      key={detection.id}
                      onClick={() => {
                        // Prevent reopening review panel for already-confirmed detections
                        if (!isConfirmedDet) setSelectedDetectionId(detection.id);
                      }}
                      className={`transition-colors group ${
                        isConfirmedDet ? 'bg-emerald-50/40 cursor-default' : 'hover:bg-teal-50/50 cursor-pointer'
                      }`}
                    >
                      {/* ── Always visible ── */}
                      <td className="py-2.5 px-3 font-mono font-bold text-teal-800">
                        {detection.full_identifier || detection.detection_id}
                      </td>

                      <td className="py-2.5 px-3 font-bold text-slate-900">{detection.object_class}</td>

                      {/* Confidence — visible on all breakpoints now */}
                      <td className="py-2.5 px-3 font-mono font-bold text-slate-900">
                        {formatConfidence(detection.confidence)}
                      </td>

                      {/* ── Desktop only ── */}
                      <td className="py-2.5 px-3 hidden lg:table-cell">
                        <span className={`inline-block px-1.5 py-0.2 rounded font-mono text-[9.5px] font-bold ${confCat.badgeClass}`}>
                          {confCat.category}
                        </span>
                      </td>

                      <td className="py-2.5 px-3 hidden md:table-cell">
                        <span className={`inline-flex items-center gap-1 px-1.5 py-0.2 rounded font-mono text-[9.5px] font-bold ${riskInfo.badgeClass}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${riskInfo.dotClass}`} />
                          {detection.risk_level}
                        </span>
                      </td>

                      <td className="py-2.5 px-3 font-mono font-bold text-slate-900 hidden lg:table-cell">
                        {detection.risk_score}
                      </td>

                      <td className="py-2.5 px-3 font-mono text-slate-800 font-medium hidden lg:table-cell">
                        {formatCoordinate(detection.latitude)}
                      </td>

                      <td className="py-2.5 px-3 font-mono text-slate-800 font-medium hidden lg:table-cell">
                        {formatCoordinate(detection.longitude)}
                      </td>

                      <td className="py-2.5 px-2 font-mono text-slate-800 font-medium hidden lg:table-cell">
                        {detection.depth != null ? `${detection.depth} m` : '—'}
                      </td>

                      <td className="py-2.5 px-3 hidden md:table-cell">
                        <span
                          className={`font-mono text-[10px] font-bold px-2 py-0.5 rounded-full uppercase border ${
                            detection.status === 'confirmed'
                              ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                              : detection.status === 'rejected'
                              ? 'bg-rose-50 text-rose-700 border-rose-200'
                              : detection.status === 'flagged'
                              ? 'bg-amber-50 text-amber-700 border-amber-200'
                              : 'bg-slate-100 text-slate-600 border-slate-200'
                          }`}
                        >
                          {detection.status}
                        </span>
                      </td>

                      {/* ── Action column ──
                          Confirmed filter: the whole column is hidden (header + cell) — nothing
                          left to action there. Any other filter (e.g. "All"): confirmed rows get
                          a disabled button so a single detection can never be re-reviewed. */}
                      {filterStatus !== 'confirmed' && (
                        <td className="py-2.5 px-3 text-right">
                          {isConfirmedDet ? (
                            <button
                              type="button"
                              disabled
                              title="Already reviewed — cannot review again"
                              className="btn secondary text-[10.5px] py-0.5 px-2 font-medium opacity-40 cursor-not-allowed whitespace-nowrap"
                            >
                              ✓ Reviewed
                            </button>
                          ) : (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedDetectionId(detection.id);
                              }}
                              className="btn secondary text-[10.5px] py-0.5 px-2 font-medium group-hover:bg-teal-600 group-hover:text-white group-hover:border-teal-600 whitespace-nowrap"
                            >
                              Review →
                            </button>
                          )}
                        </td>
                      )}
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};