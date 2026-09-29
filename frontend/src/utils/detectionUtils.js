/**
 * detectionUtils.js
 * Unified detection normalization, confidence categorization, and risk assessment
 * Ensures data consistency across Sonar Analysis, Detection Review, and Map views.
 */

export const getConfidenceCategory = (confidence) => {
  const n = Number(confidence || 0);
  const pct = n <= 1 ? n * 100 : n;

  if (pct >= 70) {
    return {
      category: 'HIGH',
      label: 'HIGH',
      color: 'teal',
      badgeClass: 'bg-teal-50 text-teal-700 border border-teal-200',
      pillClass: 'bg-teal-500 text-white',
      warning: null,
    };
  }

  if (pct >= 40) {
    return {
      category: 'MEDIUM',
      label: 'MEDIUM',
      color: 'amber',
      badgeClass: 'bg-amber-50 text-amber-700 border border-amber-200',
      pillClass: 'bg-amber-500 text-white',
      warning: null,
    };
  }

  return {
    category: 'LOW',
    label: 'LOW',
    color: 'rose',
    badgeClass: 'bg-rose-50 text-rose-700 border border-rose-200',
    pillClass: 'bg-rose-500 text-white',
    warning: 'Low confidence — Human verification required',
  };
};

export const getRiskBadgeStyle = (riskLevel) => {
  const level = String(riskLevel || 'LOW').toUpperCase();
  if (level === 'HIGH') {
    return {
      level: 'HIGH',
      badgeClass: 'bg-rose-50 text-rose-700 border border-rose-200',
      dotClass: 'bg-rose-500',
      textColor: '#e11d48',
    };
  }
  if (level === 'MEDIUM') {
    return {
      level: 'MEDIUM',
      badgeClass: 'bg-amber-50 text-amber-700 border border-amber-200',
      dotClass: 'bg-amber-500',
      textColor: '#d97706',
    };
  }
  return {
    level: 'LOW',
    badgeClass: 'bg-emerald-50 text-emerald-700 border border-emerald-200',
    dotClass: 'bg-emerald-500',
    textColor: '#059669',
  };
};

export const formatConfidence = (value) => {
  const n = Number(value || 0);
  return `${(n <= 1 ? n * 100 : n).toFixed(1)}%`;
};

export const formatCoordinate = (value, digits = 5) => {
  if (value === null || value === undefined || value === '' || Number.isNaN(Number(value))) {
    return '—';
  }
  return Number(value).toFixed(digits);
};

export const normalizeDetection = (det, surveyMeta = {}) => {
  if (!det) return null;

  const rawId = det.id ?? det.detection_id ?? '';
  const surveyIdRaw =
    det.survey_id ?? det.surveyId ?? surveyMeta.survey_id ?? surveyMeta.id ?? null;
  const survey_code =
    det.survey_code ||
    (surveyIdRaw != null && !isNaN(Number(surveyIdRaw))
      ? `SURV-${String(surveyIdRaw).padStart(3, '0')}`
      : typeof surveyIdRaw === 'string' && surveyIdRaw.startsWith('SURV')
      ? surveyIdRaw
      : 'SURV-001');

  let detection_id = det.detection_id;
  if (!detection_id) {
    const sIdx = det.survey_detection_index ?? det.surveyDetectionIndex;
    if (sIdx != null) {
      detection_id = `DET-${String(sIdx).padStart(3, '0')}`;
    } else if (rawId && typeof rawId === 'number') {
      detection_id = `DET-${String(rawId).padStart(3, '0')}`;
    } else if (typeof rawId === 'string' && rawId.startsWith('DET')) {
      detection_id = rawId.includes('-') ? rawId : `DET-${rawId.slice(3).padStart(3, '0')}`;
    } else {
      detection_id = `DET-${String(rawId || '1').padStart(3, '0')}`;
    }
  } else if (!detection_id.includes('-') && detection_id.startsWith('DET')) {
    detection_id = `DET-${detection_id.slice(3).padStart(3, '0')}`;
  }

  const full_identifier =
    det.full_identifier || `${survey_code} / ${detection_id}`;

  const id = det.id ?? detection_id;

  const object_class =
    det.object_class || det.objectClass || det.class || 'Unknown anomaly';

  const rawConf = Number(det.confidence ?? 0);
  const confPct = rawConf <= 1 ? rawConf * 100 : rawConf;
  const confCat = getConfidenceCategory(rawConf);

  // Survey metadata fallback so values already existing in upload metadata are never "Unavailable"
  const latitude =
    det.latitude != null
      ? Number(det.latitude)
      : det.lat != null
      ? Number(det.lat)
      : surveyMeta.latitude != null && surveyMeta.latitude !== ''
      ? Number(surveyMeta.latitude)
      : null;

  const longitude =
    det.longitude != null
      ? Number(det.longitude)
      : det.lng != null
      ? Number(det.lng)
      : surveyMeta.longitude != null && surveyMeta.longitude !== ''
      ? Number(surveyMeta.longitude)
      : null;

  const depth =
    det.depth != null
      ? Number(det.depth)
      : det.depthMeters != null
      ? Number(det.depthMeters)
      : surveyMeta.depth != null && surveyMeta.depth !== ''
      ? Number(surveyMeta.depth)
      : null;

  // Risk score & level
  let risk_score =
    det.risk_score != null
      ? Number(det.risk_score)
      : det.riskScore != null
      ? Number(det.riskScore)
      : null;

  let risk_level = det.risk_level
    ? String(det.risk_level).toUpperCase()
    : det.risk
    ? String(det.risk).toUpperCase()
    : null;

  if (risk_score == null) {
    const isDangerous = ['mine', 'ghost', 'wreck', 'aircraft'].some((w) =>
      object_class.toLowerCase().includes(w)
    );
    const base = isDangerous ? 75 : 45;
    risk_score = Math.min(100, Math.max(10, Math.round(base * 0.6 + confPct * 0.4)));
  }

  if (!risk_level) {
    risk_level = risk_score >= 70 ? 'HIGH' : risk_score >= 40 ? 'MEDIUM' : 'LOW';
  }

  // Bounding box calculations
  let bbox = det.bbox;
  let bbox_width_px = det.bbox_width_px ?? det.bboxWidthPx ?? null;
  let bbox_height_px = det.bbox_height_px ?? det.bboxHeightPx ?? null;
  let bbox_x = det.bbox_x ?? 0;
  let bbox_y = det.bbox_y ?? 0;

  if (Array.isArray(bbox) && bbox.length >= 4) {
    const [x1, y1, x2, y2] = bbox.map(Number);
    bbox_x = x1;
    bbox_y = y1;
    if (bbox_width_px == null) bbox_width_px = Math.max(0, Math.round(x2 - x1));
    if (bbox_height_px == null) bbox_height_px = Math.max(0, Math.round(y2 - y1));
  }

  const status = String(
    det.status || det.cleanup_status || 'pending'
  ).toLowerCase();

  return {
    ...det,
    id,
    detection_id,
    survey_code,
    full_identifier,
    object_class,
    objectClass: object_class,
    class: object_class,
    confidence: Math.round(confPct * 10) / 10,
    confidenceRaw: rawConf <= 1 ? rawConf : rawConf / 100,
    confidence_category: confCat.category,
    confidenceCategory: confCat.category,
    confidenceCategoryDetails: confCat,
    risk_score: Math.round(risk_score * 10) / 10,
    riskScore: Math.round(risk_score * 10) / 10,
    risk_level,
    risk: risk_level.toLowerCase(),
    latitude,
    lat: latitude,
    longitude,
    lng: longitude,
    depth,
    depthMeters: depth,
    bbox,
    bbox_x,
    bbox_y,
    bbox_width_px,
    bboxWidthPx: bbox_width_px,
    bbox_height_px,
    bboxHeightPx: bbox_height_px,
    status,
    survey_id:
      det.survey_id ?? det.surveyId ?? surveyMeta.survey_id ?? surveyMeta.id ?? null,
    surveyId:
      det.survey_id ?? det.surveyId ?? surveyMeta.survey_id ?? surveyMeta.id ?? null,
    image_id: det.image_id ?? det.imageId ?? null,
  };
};

export const calculateOverallRisk = (detections = []) => {
  if (!detections || detections.length === 0) {
    return {
      overallRiskScore: 0,
      overallRiskLevel: 'LOW',
      highCount: 0,
      mediumCount: 0,
      lowCount: 0,
      summary: 'No targets identified — clean acoustic profile.',
    };
  }

  let highCount = 0;
  let mediumCount = 0;
  let lowCount = 0;
  let maxScore = 0;

  detections.forEach((d) => {
    const score = Number(d.risk_score || d.riskScore || 0);
    const level = String(d.risk_level || d.risk || '').toUpperCase();
    if (score > maxScore) maxScore = score;
    if (level === 'HIGH' || score >= 70) highCount++;
    else if (level === 'MEDIUM' || score >= 40) mediumCount++;
    else lowCount++;
  });

  let overallRiskLevel = 'LOW';
  if (highCount > 0 || maxScore >= 70) {
    overallRiskLevel = 'HIGH';
  } else if (mediumCount > 0 || maxScore >= 40) {
    overallRiskLevel = 'MEDIUM';
  }

  const summary =
    overallRiskLevel === 'HIGH'
      ? `High risk identified — ${highCount} high-priority anomaly detected. Immediate verification recommended.`
      : overallRiskLevel === 'MEDIUM'
      ? `Moderate risk detected — ${mediumCount} anomaly requiring human verification.`
      : `Low operational risk — all targets within low-hazard threshold.`;

  return {
    overallRiskScore: Math.round(maxScore * 10) / 10,
    overallRiskLevel,
    highCount,
    mediumCount,
    lowCount,
    summary,
  };
};
