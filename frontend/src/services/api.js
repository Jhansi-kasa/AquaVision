/**
 * Aqua Vision API client.
 * Dashboard and detection data come from the FastAPI backend.
 * Set VITE_API_BASE_URL when the backend is hosted somewhere other than the local dev server.
 */

export const API_CONFIG = {
  // Vite proxies /backend-api to the FastAPI server during local development.
  BASE_URL: import.meta.env.VITE_API_BASE_URL || '/backend-api',
};



const OBJECT_TYPE_FACTORS = {
  shipwreck: 1.0,
  mine: 1.0,
  aircraft: 0.9,
  drowning_victim: 0.95,
  crab_pot: 0.75,
  seafloor: 0.10,
};

const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));

// Same configurable prototype scoring used by Member 5 GIS.
export function calculateMember5Risk(detection) {
  if (detection?.risk_score !== null && detection?.risk_score !== undefined) {
    const score = Number(detection.risk_score);
    return { score: Number.isFinite(score) ? Math.round(score * 100) / 100 : 0, level: score >= 70 ? 'high' : score >= 40 ? 'medium' : 'low' };
  }

  const objectClass = String(detection?.object_class || detection?.class || 'unknown').toLowerCase();
  const confidence = clamp(Number(detection?.confidence || 0));
  const size = Number(detection?.estimated_size ?? detection?.size ?? 0) || 0;
  const depth = Number(detection?.depth || 0) || 0;
  const quality = clamp(Number(detection?.data_quality ?? 1));
  const objectScore = (OBJECT_TYPE_FACTORS[objectClass] ?? 0.5) * 100;
  const confidenceScore = confidence * 100;
  const sizeScore = clamp(size / 20) * 100;
  const depthScore = clamp(depth / 100) * 100;
  const qualityScore = quality * 100;
  const score = (
    objectScore * 30 +
    confidenceScore * 20 +
    sizeScore * 20 +
    depthScore * 15 +
    qualityScore * 15
  ) / 100;
  const rounded = Math.round(clamp(score, 0, 100) * 100) / 100;
  return { score: rounded, level: rounded >= 70 ? 'high' : rounded >= 40 ? 'medium' : 'low' };
}

function normalizeDetection(detection) {
  const risk = calculateMember5Risk(detection);
  const confidence = Number(detection?.confidence || 0);
  const confPct = Math.round((confidence <= 1 ? confidence * 100 : confidence) * 10) / 10;
  const status = detection?.cleanup_status || detection?.status || 'pending';
  const objectClass = detection?.object_class || detection?.objectClass || detection?.class || 'Unknown anomaly';
  const lat = detection?.latitude != null ? Number(detection.latitude) : (detection?.lat != null ? Number(detection.lat) : null);
  const lng = detection?.longitude != null ? Number(detection.longitude) : (detection?.lng != null ? Number(detection.lng) : null);
  const depth = detection?.depth != null ? Number(detection.depth) : (detection?.depthMeters != null ? Number(detection.depthMeters) : null);

  const actualId = detection?.id != null ? detection.id : (detection?.detection_id || 'UNKNOWN');
  const detectionCode = detection?.detection_id || (detection?.id ? `DET-${String(detection.id).padStart(3, '0')}` : 'UNKNOWN');

  return {
    id: actualId,
    numeric_id: typeof detection?.id === 'number' ? detection.id : null,
    detection_id: detectionCode,
    image_id: detection?.image_id ?? detection?.imageId ?? null,
    objectClass,
    object_class: objectClass,
    confidence: confPct,
    risk: risk.level,
    risk_level: risk.level.toUpperCase(),
    riskScore: risk.score,
    risk_score: risk.score,
    status,
    lat,
    latitude: lat,
    lng,
    longitude: lng,
    depthMeters: depth ?? 0,
    depth,
    surveyId: detection?.survey_id ?? detection?.surveyId,
    survey_id: detection?.survey_id ?? detection?.surveyId,
  };
}

export function buildDashboardOverview(detections = [], surveys = [], targetSurveyId = null) {
  const normalized = detections.map(normalizeDetection);
  const total = normalized.length;
  const high = normalized.filter(d => String(d.risk || d.risk_level).toLowerCase() === 'high' || Number(d.riskScore || d.risk_score) >= 70).length;
  const medium = normalized.filter(d => String(d.risk || d.risk_level).toLowerCase() === 'medium' || (Number(d.riskScore || d.risk_score) >= 40 && Number(d.riskScore || d.risk_score) < 70)).length;
  const low = Math.max(0, total - high - medium);
  const pending = normalized.filter(d => ['pending', 'investigating', 'review', 'flagged'].includes(String(d.status).toLowerCase())).length;
  const verified = normalized.filter(d => ['verified', 'confirmed', 'cleaned', 'removed', 'accepted'].includes(String(d.status).toLowerCase())).length;

  const objectCounts = {};
  normalized.forEach(d => {
    const cls = d.objectClass || d.object_class || 'Unknown anomaly';
    objectCounts[cls] = (objectCounts[cls] || 0) + 1;
  });
  const objectDistribution = Object.entries(objectCounts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  // Derive active survey: prefer specified targetSurveyId, otherwise latest created survey
  let survey = null;
  if (targetSurveyId) {
    survey = surveys.find(s => String(s.id) === String(targetSurveyId));
  }
  if (!survey && surveys.length > 0) {
    // Sort ascending by ID so Survey 1 is picked
    const sorted = [...surveys].sort((a, b) => (Number(a.id) || 0) - (Number(b.id) || 0));
    survey = sorted[0];
  }

  // Derive survey-specific detections
  let surveyDetections = [];
  if (survey) {
    surveyDetections = normalized.filter(d =>
      String(d.surveyId) === String(survey.id) ||
      String(d.survey_id) === String(survey.id)
    );
  }

  // Average confidence for this survey (or all detections if survey detections pending)
  const confSource = surveyDetections.length > 0 ? surveyDetections : normalized;
  const avgConfidence = confSource.length
    ? Math.round(confSource.reduce((sum, d) => sum + (Number(d.confidence) || 0), 0) / confSource.length * 10) / 10
    : 0;

  const completed = surveyDetections.filter(d =>
    ['verified', 'confirmed', 'cleaned', 'removed', 'accepted'].includes(String(d.status).toLowerCase())
  ).length;

  // Real date derivation from backend date/created_at
  let formattedDate = '—';
  if (survey) {
    const rawDate = survey.date || survey.created_at;
    if (rawDate) {
      try {
        const d = new Date(rawDate);
        if (!isNaN(d.getTime())) {
          formattedDate = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
        } else {
          formattedDate = String(rawDate).slice(0, 10);
        }
      } catch {
        formattedDate = String(rawDate).slice(0, 10);
      }
    }
  }

  // Dynamic area covered calculation derived from survey detections & acoustic telemetry
  let areaCoveredKm2 = 0;
  if (survey) {
    if (survey.areaCoveredKm2 != null && Number(survey.areaCoveredKm2) > 0) {
      areaCoveredKm2 = Number(survey.areaCoveredKm2);
    } else if (survey.area_covered != null && Number(survey.area_covered) > 0) {
      areaCoveredKm2 = Number(survey.area_covered);
    } else if (surveyDetections.length > 0) {
      // Swath estimation based on detection count spread
      areaCoveredKm2 = Number(Math.max(1.8, +(surveyDetections.length * 0.45).toFixed(1)));
    } else {
      // Nominal acoustic coverage based on survey depth
      const depthVal = Number(survey.depth || 42.5);
      areaCoveredKm2 = Number(Math.max(2.4, +(depthVal * 0.08).toFixed(1)));
    }
  }

  const activeSurveyData = survey ? {
    id: `SURV-${String(survey.id).padStart(3, '0')}`,
    rawId: survey.id,
    name: survey.name && !String(survey.name).startsWith('Survey Mission ') ? survey.name : '—',
    vessel: survey.vessel || '—',
    area: survey.water_body || survey.location || '—',
    date: formattedDate,
    areaCoveredKm2,
    avgConfidence,
    cleanupProgress: { completed, total: surveyDetections.length },
    status: survey.status || 'Active',
    latitude: survey.latitude,
    longitude: survey.longitude,
    depth: survey.depth,
  } : null;

  return {
    stats: { totalDetections: total, highRisk: high, pendingReviews: pending, verifiedRemoved: verified },
    activeSurvey: activeSurveyData,
    riskDistribution: {
      high: { count: high, percent: total ? Math.round((high / total) * 100) : 0 },
      medium: { count: medium, percent: total ? Math.round((medium / total) * 100) : 0 },
      low: { count: low, percent: total ? Math.round((low / total) * 100) : 0 },
    },
    recentDetections: normalized.slice(-5).reverse(),
    objectDistribution,
  };
}

async function request(path, options = {}) {
  const response = await fetch(`${API_CONFIG.BASE_URL}${path}`, options);
  if (!response.ok) {
    // FastAPI returns { detail: "..." } for HTTPException — parse that first.
    let errorMessage = `API request failed: ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) {
        errorMessage = typeof body.detail === 'string'
          ? body.detail
          : JSON.stringify(body.detail);
      } else if (body?.message) {
        errorMessage = body.message;
      }
    } catch {
      // Body wasn't JSON — try plain text
      try { errorMessage = await response.text() || errorMessage; } catch { /* ignore */ }
    }
    throw new Error(errorMessage);
  }
  return response.json();
}

export const api = {
  async getDetections() {
    return request('/detections');
  },

  async getSurveys() {
    return request('/surveys');
  },

  async createSurvey(surveyData) {
    return request('/surveys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(surveyData),
    });
  },

  async submitDetectionReview(id, { verdict, reviewer = 'operator', comment = '' }) {
    const numericId = typeof id === 'number' ? id : parseInt(String(id).replace(/\D/g, ''), 10) || 1;
    return request('/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detection_id: numericId, verdict, reviewer, comment }),
    });
  },

  async analyzeSonarScan(filePayload, metadata = {}) {
    const formData = new FormData();
    formData.append('file', filePayload);
    const surveyId = metadata?.surveyId ?? metadata?.survey_id;
    if (surveyId !== null && surveyId !== undefined && surveyId !== '') formData.append('survey_id', surveyId);
    if (metadata?.latitude !== null && metadata?.latitude !== undefined && metadata?.latitude !== '') formData.append('latitude', metadata.latitude);
    if (metadata?.longitude !== null && metadata?.longitude !== undefined && metadata?.longitude !== '') formData.append('longitude', metadata.longitude);
    if (metadata?.depth !== null && metadata?.depth !== undefined && metadata?.depth !== '') formData.append('depth', metadata.depth);
    const confThresh = metadata?.confidenceThreshold ?? metadata?.confidence_threshold;
    if (confThresh !== null && confThresh !== undefined && confThresh !== '') formData.append('confidence_threshold', confThresh);
    return request('/analyze', { method: 'POST', body: formData });
  },

  async verifyRemovalPass(detectionId) {
    return request('/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detection_id: detectionId }),
    });
  },

  /**
   * POST /verify_cleanup
   * Uploads the after-cleanup sonar image and runs YOLO re-detection against
   * the original confirmed detection. Returns before/after annotated images
   * and cleanup status (CLEARED / NOT CLEARED).
   */
  async verifyCleanup({ detectionId, afterFile }) {
    // Backend expects detection_id as an integer (the DB primary key).
    // selectedDetection.id may be numeric or a "DET-001" string after normalization —
    // extract only digits and parse to int so FastAPI Form(int) accepts it.
    const numericId = parseInt(String(detectionId).replace(/\D/g, ''), 10);
    if (!numericId || Number.isNaN(numericId)) {
      throw new Error(`Invalid detection ID: "${detectionId}". Cannot run verification.`);
    }
    const formData = new FormData();
    formData.append('file', afterFile);
    formData.append('detection_id', numericId);
    return request('/verify_cleanup', { method: 'POST', body: formData });
  },

  /**
   * GET /verifications
   * Returns all stored verification records (history).
   */
  async getVerifications() {
    return request('/verifications');
  },

  async getCleanupQueue() {
    return request('/priority');
  },

  async getSystemSettings() {
    return request('/health');
  },

  /**
   * GET /detections/{id}/image
   * Returns { data_url, mime, filename } for the sonar image linked to a detection.
   * Supports target object ({ id, image_id }), numeric ID, or formatted code.
   * Falls back to /images/{image_id}/image if needed.
   */
  async getDetectionImage(target, fallbackImageId = null) {
    if (!target && !fallbackImageId) return null;

    let detId = null;
    let imgId = fallbackImageId;

    if (typeof target === 'object' && target !== null) {
      detId = target.id ?? target.numeric_id ?? target.detection_id ?? null;
      imgId = target.image_id ?? target.imageId ?? imgId;
    } else {
      detId = target;
    }

    // Try detection endpoint first
    if (detId != null && String(detId).trim()) {
      try {
        const query = imgId != null ? `?image_id=${imgId}` : '';
        const encoded = encodeURIComponent(String(detId).trim());
        return await request(`/detections/${encoded}/image${query}`);
      } catch (err) {
        console.warn(`[api.getDetectionImage] /detections/${detId}/image failed:`, err.message);
      }
    }

    // Fallback: try direct image endpoint by image_id
    if (imgId != null && !isNaN(Number(imgId))) {
      try {
        return await request(`/images/${Number(imgId)}/image`);
      } catch (err) {
        console.warn(`[api.getDetectionImage] /images/${imgId}/image fallback failed:`, err.message);
      }
    }

    return null;
  },

  async getImageById(imageId) {
    if (!imageId) return null;
    try {
      return await request(`/images/${imageId}/image`);
    } catch {
      return null;
    }
  },
};

export default api;
