import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useToast } from '../../context/ToastContext';
import { api } from '../../services/api';
import {
  normalizeDetection,
  getRiskBadgeStyle,
  formatCoordinate,
  formatConfidence,
} from '../../utils/detectionUtils';

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const CONFIRMED_STATUSES = new Set(['confirmed', 'verified', 'accepted', 'cleaned']);

function isConfirmed(det) {
  return CONFIRMED_STATUSES.has(String(det?.status || det?.cleanup_status || '').toLowerCase());
}

function CleanupStatusBadge({ status }) {
  const upper = String(status || '').toUpperCase();
  if (upper === 'CLEARED' || upper === 'CLEANED') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />CLEARED
      </span>
    );
  }
  if (upper === 'NOT CLEARED' || upper === 'STILL_PRESENT') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-rose-50 text-rose-700 border border-rose-200">
        <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />NOT CLEARED
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-50 text-amber-700 border border-amber-200">
      <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />PENDING
    </span>
  );
}

function VerifyStatusBadge({ result }) {
  if (!result) {
    return (
      <span className="inline-block px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-slate-100 text-slate-500 border border-slate-200">
        AWAITING UPLOAD
      </span>
    );
  }
  if (result === 'potentially_removed') {
    return (
      <span className="inline-block px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
        ✓ DONE
      </span>
    );
  }
  return (
    <span className="inline-block px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-rose-50 text-rose-700 border border-rose-200">
      ✕ NOT CLEARED
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

export const VerifyPage = ({ confirmedDetections = [], onNavigate }) => {
  const { showToast } = useToast();
  const fileInputRef = useRef(null);

  // Normalized confirmed detections eligible for verification
  const [targets, setTargets] = useState([]);

  // Selected detection ID
  const [selectedId, setSelectedId] = useState(null);

  // After-cleanup upload state
  const [afterFile, setAfterFile] = useState(null);
  const [afterPreview, setAfterPreview] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [beforeImageSrc, setBeforeImageSrc] = useState(null);

  // Verification result from API
  const [verifyResult, setVerifyResult] = useState(null);

  // History of completed verifications
  const [history, setHistory] = useState([]);

  // ── Normalize and set targets when prop changes ──────────────────────────
  useEffect(() => {
    const normalized = (confirmedDetections || [])
      .map((d) => normalizeDetection(d))
      .filter(Boolean)
      .filter(isConfirmed);

    setTargets(normalized);
    // Auto-select first if nothing is selected
    if (normalized.length > 0) {
      setSelectedId((prev) => {
        const stillExists = normalized.find((d) => String(d.id) === String(prev));
        return stillExists ? prev : normalized[0].id;
      });
    }
  }, [confirmedDetections]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Load verification history from backend ────────────────────────────────
  useEffect(() => {
    api.getVerifications().then((data) => {
      if (Array.isArray(data)) setHistory(data);
    }).catch(() => {});
  }, []);

  // ── Fetch original detection image when selectedId changes ────────────────
  useEffect(() => {
    if (!selectedId) {
      setBeforeImageSrc(null);
      return;
    }
    const currentTarget = targets.find((d) => String(d.id) === String(selectedId));
    let cancelled = false;
    api.getDetectionImage(currentTarget || selectedId, currentTarget?.image_id).then((res) => {
      if (!cancelled) setBeforeImageSrc(res?.data_url || null);
    }).catch(() => {
      if (!cancelled) setBeforeImageSrc(null);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedId, targets]);

  const selectedDetection = targets.find((d) => String(d.id) === String(selectedId)) || null;

  const historyEntry = history.find(
    (h) => String(h.detection_id) === String(selectedDetection?.id)
  ) || null;

  // ── File handler ──────────────────────────────────────────────────────────
  const handleFile = useCallback((file) => {
    if (!file) return;
    setAfterFile(file);
    setVerifyResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setAfterPreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  // ── Select detection ──────────────────────────────────────────────────────
  const handleSelect = (id) => {
    setSelectedId(id);
    setBeforeImageSrc(null);
    setAfterFile(null);
    setAfterPreview(null);
    setVerifyResult(null);
  };

  // ── Run verification ──────────────────────────────────────────────────────
  const handleRunVerify = async () => {
    if (!selectedDetection) {
      showToast({ type: 'warning', message: 'Select a confirmed detection first.' });
      return;
    }
    if (!afterFile) {
      showToast({ type: 'warning', message: 'Upload a post-cleanup sonar image first.' });
      return;
    }

    setIsVerifying(true);
    try {
      // Debug: log the exact ID and type being sent — visible in browser DevTools console
      console.debug('[VerifyCleanup] detectionId:', selectedDetection.id, 'type:', typeof selectedDetection.id, 'status:', selectedDetection.status);

      const result = await api.verifyCleanup({
        detectionId: selectedDetection.id,
        afterFile,
      });

      setVerifyResult(result);

      // Update history
      setHistory((prev) => [
        result,
        ...prev.filter((h) => String(h.detection_id) !== String(selectedDetection.id)),
      ]);

      // If cleared, update local status
      if (result.result === 'potentially_removed') {
        setTargets((prev) =>
          prev.map((d) =>
            String(d.id) === String(selectedDetection.id) ? { ...d, status: 'cleaned' } : d
          )
        );
        showToast({
          type: 'success',
          message: `✅ ${selectedDetection.detection_id} — CLEANUP CONFIRMED. Object no longer detected.`,
        });
      } else {
        showToast({
          type: 'warning',
          message: `⚠️ ${selectedDetection.detection_id} — Object STILL DETECTED in after-cleanup scan.`,
        });
      }
    } catch (err) {
      showToast({ type: 'error', message: err.message || 'Verification failed — check backend connection.' });
    } finally {
      setIsVerifying(false);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // EMPTY STATE
  // ─────────────────────────────────────────────────────────────────────────
  if (targets.length === 0) {
    return (
      <section className="animate-page-fade space-y-4 bg-[#EFF8FB] min-h-screen -m-6 p-6">
        <div className="border-b border-slate-100 pb-2">
          <h1 className="text-[18px] font-bold text-slate-800">Cleanup Verification</h1>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-8 text-center max-w-md mx-auto my-6">
          <div className="w-10 h-10 rounded-full bg-teal-50 text-teal-600 mx-auto flex items-center justify-center mb-3">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <h2 className="text-[14px] font-bold text-slate-800 mb-1.5">No Confirmed Detections Awaiting Verification</h2>
          <p className="text-[12px] text-slate-500 mb-5">
            No confirmed detections are awaiting cleanup verification.
            <br />
            Confirm detections on the Human Verification page first.
          </p>
          <button
            onClick={() => onNavigate?.('review')}
            className="btn primary mx-auto text-[12px] py-1.5 px-4"
          >
            Go to Human Verification →
          </button>
        </div>
      </section>
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // MAIN VIEW
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <section className="animate-page-fade space-y-5 bg-[#EFF8FB] min-h-screen -m-6 p-6">

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-100">
        <div>
          <p className="text-[16px] text-slate-900">
            Upload post-cleanup sonar scans — AI re-detection determines if the object is still present.
          </p>
        </div>
        <span className="font-mono text-[10.5px] text-teal-700 bg-teal-50 border border-teal-200 px-2.5 py-1 rounded-lg">
          {targets.length} confirmed target{targets.length !== 1 ? 's' : ''} eligible
        </span>
      </div>

      {/* Detection Selector — dropdown */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4">
        <label className="block text-[11px] font-mono font-bold text-slate-700 uppercase tracking-wider mb-2">
          Select Confirmed Detection to Verify
        </label>
        <select
          value={selectedId ?? ''}
          onChange={(e) => handleSelect(e.target.value)}
          className="w-full border border-slate-300 rounded-lg px-3 py-2.5 text-[13px] font-medium text-slate-900 bg-white focus:outline-none focus:ring-2 focus:ring-teal-400 focus:border-teal-400 appearance-none cursor-pointer"
        >
          {targets.map((det) => {
            const hist = history.find((h) => String(h.detection_id) === String(det.id));
            const cleared = det.status === 'cleaned' || hist?.result === 'potentially_removed';
            const label = `${det.full_identifier || det.detection_id} — ${det.object_class}${cleared ? ' ✓ Cleared' : ''}`;
            return (
              <option key={det.id} value={det.id}>{label}</option>
            );
          })}
        </select>
      </div>

      {/* Detail panel for selected detection */}
      {selectedDetection && (
        <div className="space-y-4">

          {/* Metadata card */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4">
            <div className="flex flex-wrap items-start justify-between gap-3 mb-3 pb-3 border-b border-slate-100">
              <div>
                <span className="text-[10px] font-mono text-slate-700 uppercase">Selected Target</span>
                <h2 className="text-[16px] font-bold text-slate-800 leading-tight">
                  {selectedDetection.full_identifier || selectedDetection.detection_id}
                </h2>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className={`font-mono text-[10.5px] font-bold px-2 py-0.5 rounded-full border uppercase ${getRiskBadgeStyle(selectedDetection.risk_level).badgeClass}`}>
                  {selectedDetection.risk_level} RISK
                </span>
                <span className="font-mono text-[10.5px] font-bold px-2 py-0.5 rounded-full border uppercase bg-emerald-50 text-emerald-700 border-emerald-200">
                  CONFIRMED
                </span>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-2.5 text-[11.5px]">
              {[
                [<span className="text-[11px] text-slate-950 font-semibold">Object Class</span>, <span className="text-slate-800 capitalize">{selectedDetection.object_class}</span>],
                [<span className="text-[11px] text-slate-950 font-semibold">Risk Score</span>, <span className="font-mono text-slate-800">{selectedDetection.risk_score} %</span>],
                [<span className="text-[11px] text-slate-950 font-semibold">Latitude</span>, <span className="font-mono text-slate-800">{formatCoordinate(selectedDetection.latitude)}</span>],
                [<span className="text-[11px] text-slate-950 font-semibold">Longitude</span>, <span className="font-mono text-slate-800">{formatCoordinate(selectedDetection.longitude)}</span>],
                [<span className="text-[11px] text-slate-950 font-semibold">Depth</span>, <span className="font-mono text-slate-800">{selectedDetection.depth != null ? `${selectedDetection.depth} m` : '—'}</span>],
                [<span className="text-[11px] text-slate-950 font-semibold">AI Confidence</span>, <span className="font-mono text-slate-800">{formatConfidence(selectedDetection.confidence)}</span>],
                [<span className="text-[11px] text-slate-950 font-semibold">Survey</span>, <span className="font-mono text-slate-800">{selectedDetection.survey_code || (selectedDetection.survey_id ? `SURV-${String(selectedDetection.survey_id).padStart(3,'0')}` : '—')}</span>],
              ].map(([label, value]) => (
                <div key={label}>
                  <div className="text-[9.5px] font-mono text-slate-400 uppercase mb-0.5">{label}</div>
                  {value}
                </div>
              ))}
            </div>
          </div>

          {/* Before / After panels */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

            {/* BEFORE */}
            <div className="bg-[#020a0f] border border-slate-700 rounded-xl p-3.5 flex flex-col gap-3 shadow-[inset_0_0_30px_rgba(45,212,196,0.06)]">
              <div className="flex items-center justify-between font-mono text-[9.5px] tracking-wider">
                <span className="text-[#00ff66] font-bold">● BEFORE · ORIGINAL DETECTION</span>
                <span className="text-slate-400">{selectedDetection.detection_id}</span>
              </div>

              <div className="w-full aspect-[4/3] rounded overflow-hidden bg-[#010607] flex items-center justify-center">
                {beforeImageSrc || verifyResult?.before_image ? (
                  <img
                    src={beforeImageSrc || verifyResult?.before_image}
                    alt="Before cleanup — original detection"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center p-5">
                    <div className="text-[#00ff66] font-mono text-[9px] mb-1">DETECTION IMAGE</div>
                    <div className="text-slate-500 font-mono text-[9px]">IMG-{selectedDetection.image_id ?? '—'}</div>
                    <div className="mt-3 text-slate-600 font-mono text-[9px]">
                      {selectedDetection.object_class} · {formatConfidence(selectedDetection.confidence)}
                    </div>
                    {selectedDetection.bbox && selectedDetection.bbox.length >= 4 && (
                      <div className="text-slate-700 font-mono text-[8.5px] mt-1">
                        bbox [{selectedDetection.bbox.map((v) => Math.round(v)).join(', ')}]
                      </div>
                    )}
                    <div className="text-slate-600 font-mono text-[8.5px] mt-2">
                      ↑ Loading original detection scan…
                    </div>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] font-mono">
                <div><span className="text-slate-500">Class: </span><span className="text-teal-300 font-bold capitalize">{selectedDetection.object_class}</span></div>
                <div><span className="text-slate-500">Conf: </span><span className="text-teal-300 font-bold">{formatConfidence(selectedDetection.confidence)}</span></div>
                <div>
                  <span className="text-slate-500">Risk: </span>
                  <span className={`font-bold ${selectedDetection.risk_level === 'HIGH' ? 'text-rose-400' : selectedDetection.risk_level === 'MEDIUM' ? 'text-amber-400' : 'text-emerald-400'}`}>
                    {selectedDetection.risk_level}
                  </span>
                </div>
                <div><span className="text-slate-500">Status: </span><span className="text-[#00ff66] font-bold">CONFIRMED</span></div>
              </div>
            </div>

            {/* AFTER */}
            <div className="bg-[#020a0f] border border-slate-700 rounded-xl p-3.5 flex flex-col gap-3 shadow-[inset_0_0_30px_rgba(45,212,196,0.06)]">
              <div className="flex items-center justify-between font-mono text-[9.5px] tracking-wider">
                <span className="text-teal-300 font-bold">◌ AFTER · CLEANUP SCAN</span>
                <span className={`font-bold ${isVerifying ? 'text-amber-400 animate-pulse' : verifyResult ? 'text-teal-300' : afterPreview ? 'text-slate-300' : 'text-slate-500'}`}>
                  {isVerifying ? 'RUNNING YOLO…' : verifyResult ? 'VERIFIED' : afterPreview ? 'LOADED' : 'AWAITING UPLOAD'}
                </span>
              </div>

              <div className="w-full aspect-[4/3] rounded overflow-hidden bg-[#010607] flex items-center justify-center">
                {verifyResult?.after_image ? (
                  <img
                    src={verifyResult.after_image}
                    alt="After cleanup — YOLO re-detection result"
                    className="w-full h-full object-contain"
                  />
                ) : afterPreview ? (
                  <img
                    src={afterPreview}
                    alt="After cleanup sonar (uploaded, pending YOLO)"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center p-5">
                    <svg viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-8 h-8 mx-auto text-slate-600 mb-2">
                      <path d="M20 6v20M12 18l8 8 8-8" strokeLinecap="round" strokeLinejoin="round" />
                      <rect x="6" y="30" width="28" height="6" rx="1.5" strokeLinecap="round" />
                    </svg>
                    <div className="text-slate-500 font-mono text-[9.5px]">NO SCAN LOADED</div>
                    <div className="text-slate-600 font-mono text-[8.5px] mt-1">Upload post-cleanup sonar below</div>
                  </div>
                )}
              </div>

              {/* Upload zone */}
              <input
                type="file"
                ref={fileInputRef}
                accept="image/*,.tiff,.tif"
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border border-dashed border-slate-600 hover:border-teal-400 hover:bg-[rgba(45,212,196,0.04)] rounded-md py-2.5 px-3 text-center cursor-pointer transition-colors"
              >
                <div className="text-[11px] font-medium text-slate-300">
                  {afterFile ? `📎 ${afterFile.name}` : 'Upload After-Cleanup Sonar'}
                </div>
                <div className="text-[9.5px] text-slate-500 font-mono mt-0.5">
                  {afterFile ? 'Click to change file' : 'Click to choose post-cleanup re-scan image'}
                </div>
              </div>

              <button
                onClick={handleRunVerify}
                disabled={isVerifying || !afterFile}
                className="btn primary w-full justify-center text-[12px] py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {isVerifying ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Running YOLO comparison…
                  </span>
                ) : 'Run Verification'}
              </button>
            </div>
          </div>

          {/* Result banner */}
          {verifyResult && (() => {
            const cleared = verifyResult.result === 'potentially_removed';
            const othersPresent = verifyResult.other_objects_present ||
              (verifyResult.other_objects_detected && verifyResult.other_objects_detected.length > 0);
            const others = verifyResult.other_objects_detected || [];

            return (
              <div className="space-y-3">
                {/* Primary result */}
                <div className={`p-4 rounded-xl border flex flex-col sm:flex-row sm:items-start gap-3 ${
                  cleared
                    ? 'bg-emerald-50 border-emerald-300'
                    : 'bg-rose-50 border-rose-300'
                }`}>
                  <div className="flex-1">
                    <div className={`font-bold text-[14px] mb-1 ${cleared ? 'text-emerald-800' : 'text-rose-800'}`}>
                      {cleared ? '✅ CLEANUP STATUS: CLEARED' : '⚠️ CLEANUP STATUS: NOT CLEARED'}
                    </div>
                    <div className={`text-[12px] font-mono ${cleared ? 'text-emerald-700' : 'text-rose-700'}`}>
                      {cleared
                        ? `Object no longer detected after cleanup.`
                        : `Object still detected in after-cleanup sonar scan.`}
                      {' '}(IoU: {verifyResult.best_iou ?? 0}, Conf: {verifyResult.confidence})
                    </div>
                    {verifyResult.timestamp && (
                      <div className="text-[10px] font-mono mt-1 text-slate-400">
                        Verified at: {new Date(verifyResult.timestamp).toLocaleString()}
                      </div>
                    )}
                  </div>
                  <div className="flex-shrink-0 pt-0.5">
                    {cleared ? (
                      <span className="inline-block px-3 py-1 rounded-full text-[11px] font-bold bg-emerald-600 text-white">
                        CLEANUP DONE
                      </span>
                    ) : (
                      <span className="inline-block px-3 py-1 rounded-full text-[11px] font-bold bg-rose-600 text-white">
                        RE-CLEAN REQUIRED
                      </span>
                    )}
                  </div>
                </div>

                {/* Secondary warning — other objects detected even though the target was cleared */}
                {cleared && othersPresent && (
                  <div className="p-3.5 rounded-xl border border-amber-300 bg-amber-50 flex flex-col sm:flex-row sm:items-start gap-3">
                    <div className="flex-1">
                      <div className="font-bold text-[13px] text-amber-800 mb-1">
                        ⚠️ Other Objects Detected in After-Cleanup Scan
                      </div>
                      <div className="text-[11.5px] text-amber-700 font-mono mb-1.5">
                        The target ({verifyResult.detection?.object_class || 'confirmed object'}) is cleared,
                        but the after-cleanup scan contains the following additional detections:
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {others.map((d, i) => (
                          <span
                            key={i}
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-100 text-amber-800 border border-amber-300"
                          >
                            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" />
                            {d.class} · {(d.confidence * 100).toFixed(1)}%
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="flex-shrink-0 pt-0.5">
                      <span className="inline-block px-2.5 py-1 rounded-full text-[10.5px] font-bold bg-amber-500 text-white">
                        NEW TARGETS
                      </span>
                    </div>
                  </div>
                )}
              </div>
            );
          })()}

          {/* Technical note */}
          <div className="text-[11px] text-slate-500 leading-relaxed p-2.5 border-l-2 border-slate-300 bg-slate-50 rounded-r-md">
            <strong>Verification method:</strong> YOLO re-detection is run on the uploaded after-cleanup sonar image using the same
            preprocessing pipeline (CLAHE + bilateral filter) as the original survey scan. The result is matched against the confirmed
            detection's bounding box using IoU ≥ 0.30. Changes in water turbidity, tide, or sonar orientation between passes may affect accuracy.
          </div>
        </div>
      )}

      {/* Verification History */}
      {history.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
          <div className="px-4 py-2.5 border-b border-slate-100 bg-slate-50">
            <span className="text-[10.5px] font-mono text-slate-500 uppercase tracking-wider">
              Verification History ({history.length})
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-[11px] font-mono text-slate-900 uppercase tracking-wider bg-slate-50">
                  {/* Always visible */}
                  <th className="py-2 px-3">Detection ID</th>
                  <th className="py-2 px-3">Result</th>
                  <th className="py-2 px-3">Object Class</th>
                  {/* Hidden on mobile, shown on md+ */}
                  <th className="py-2 px-3 hidden md:table-cell">Survey</th>
                  <th className="py-2 px-3 hidden md:table-cell">Confidence</th>
                  <th className="py-2 px-3 hidden md:table-cell">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-[11px]">
                {history.map((h, idx) => {
                  const det = targets.find((d) => String(d.id) === String(h.detection_id));
                  const label = det
                    ? (det.full_identifier || det.detection_id)
                    : (h.detection?.detection_id || `DET-${String(h.detection_id).padStart(3,'0')}`);
                  const surveyLabel = det
                    ? (det.survey_code || (det.survey_id ? `SURV-${String(det.survey_id).padStart(3,'0')}` : '—'))
                    : (h.detection?.survey_code || '—');
                  const classLabel = det?.object_class || h.detection?.object_class || '—';

                  return (
                    <tr key={h.verification_id ?? h.id ?? idx} className="hover:bg-slate-50">
                      {/* Always visible */}
                      <td className="py-2 px-3 font-mono font-semibold text-teal-700">{label}</td>
                      <td className="py-2 px-3"><VerifyStatusBadge result={h.result} /></td>
                      <td className="py-2 px-3 text-slate-700 capitalize">{classLabel}</td>
                      {/* Hidden on mobile */}
                      <td className="py-2 px-3 font-mono text-slate-600 hidden md:table-cell">{surveyLabel}</td>
                      <td className="py-2 px-3 font-mono text-slate-600 hidden md:table-cell">
                        {h.confidence != null ? `${(h.confidence * 100).toFixed(0)}%` : '—'}
                      </td>
                      <td className="py-2 px-3 font-mono text-slate-400 text-[10px] hidden md:table-cell">
                        {h.timestamp ? new Date(h.timestamp).toLocaleString() : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
};