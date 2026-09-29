import React, { useState, useMemo, useEffect } from 'react';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  Popup,
  useMap,
} from 'react-leaflet';
import { useToast } from '../../context/ToastContext';

// ─────────────────────────────────────────────────────────────────────────────
// Haversine Distance
// ─────────────────────────────────────────────────────────────────────────────
const haversineDistanceKm = (lat1, lon1, lat2, lon2) => {
  if (
    lat1 == null ||
    lon1 == null ||
    lat2 == null ||
    lon2 == null ||
    isNaN(lat1) ||
    isNaN(lon1) ||
    isNaN(lat2) ||
    isNaN(lon2)
  ) {
    return 0;
  }

  const R = 6371;

  const dLat = (lat2 - lat1) * (Math.PI / 180);
  const dLon = (lon2 - lon1) * (Math.PI / 180);

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * (Math.PI / 180)) *
      Math.cos(lat2 * (Math.PI / 180)) *
      Math.sin(dLon / 2) ** 2;

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
};

// ─────────────────────────────────────────────────────────────────────────────
// Compass bearing from point 1 to point 2 (0° = North, 90° = East)
// ─────────────────────────────────────────────────────────────────────────────
const calculateBearing = (lat1, lon1, lat2, lon2) => {
  if (
    lat1 == null ||
    lon1 == null ||
    lat2 == null ||
    lon2 == null ||
    isNaN(lat1) ||
    isNaN(lon1) ||
    isNaN(lat2) ||
    isNaN(lon2)
  ) {
    return 0;
  }

  const toRad = (deg) => (deg * Math.PI) / 180;
  const toDeg = (rad) => (rad * 180) / Math.PI;

  const dLon = toRad(lon2 - lon1);
  const y = Math.sin(dLon) * Math.cos(toRad(lat2));
  const x =
    Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) -
    Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLon);

  const bearing = toDeg(Math.atan2(y, x));

  return (bearing + 360) % 360;
};

// ─────────────────────────────────────────────────────────────────────────────
// Local fallback route planner (same algorithm as the Python planner)
//   1. Vessel start exists  -> route begins at the vessel (object or no object).
//      No vessel start      -> try candidate starting detections and keep the
//                              shortest total route (that first stop is "S").
//   2. Distance decides the next area; priority breaks near-ties.
//   3. Everything within CLUSTER_DISTANCE_KM of the seed is one cleanup area.
//   4. Inside an area: HIGH tier first, nearest stop inside each tier.
// ─────────────────────────────────────────────────────────────────────────────
const CLUSTER_DISTANCE_KM = 0.5;
const AREA_TIE_KM = 0.1;
const INSIDE_TIE_KM = 0.01;
const MAX_START_CANDIDATES = 50;

const riskRank = (level) => (level === 'high' ? 1 : level === 'medium' ? 2 : 3);

const byPriority = (a, b) =>
  riskRank(a.riskLevel) - riskRank(b.riskLevel) ||
  b.riskScore - a.riskScore ||
  b.confidence - a.confidence;

// Run the planner once from a given origin.
//   hasStart = true  -> first leg is measured from the origin (vessel)
//   hasStart = false -> route starts AT the first stop (first leg = 0)
const runLocalRoute = (targets, start, hasStart) => {
  const remaining = [...targets];
  const ordered = [];
  let prev = hasStart ? { lat: start.lat, lng: start.lng } : null;
  let pos = { lat: start.lat, lng: start.lng };
  let clusterNo = 0;
  let total = 0;

  while (remaining.length) {
    const measured = remaining.map((t) => ({
      t,
      d: haversineDistanceKm(pos.lat, pos.lng, t.lat, t.lng),
    }));
    const nearest = Math.min(...measured.map((m) => m.d));
    const candidates = measured
      .filter((m) => m.d <= nearest + AREA_TIE_KM)
      .sort((a, b) => byPriority(a.t, b.t) || a.d - b.d);
    const seed = candidates[0].t;
    const tie = candidates[0].d > nearest + 1e-9;

    const cluster = remaining.filter(
      (t) =>
        t === seed ||
        haversineDistanceKm(seed.lat, seed.lng, t.lat, t.lng) <=
          CLUSTER_DISTANCE_KM
    );
    cluster.forEach((t) => remaining.splice(remaining.indexOf(t), 1));
    clusterNo += 1;

    const pending = [...cluster];
    let first = true;

    while (pending.length) {
      const bestRank = Math.min(...pending.map((t) => riskRank(t.riskLevel)));
      const next = pending
        .filter((t) => riskRank(t.riskLevel) === bestRank)
        .map((t) => ({
          t,
          d: haversineDistanceKm(pos.lat, pos.lng, t.lat, t.lng),
        }))
        .sort(
          (a, b) =>
            Math.round(a.d / INSIDE_TIE_KM) - Math.round(b.d / INSIDE_TIE_KM) ||
            b.t.riskScore - a.t.riskScore
        )[0].t;

      pending.splice(pending.indexOf(next), 1);

      const segment = prev
        ? haversineDistanceKm(prev.lat, prev.lng, next.lat, next.lng)
        : 0;
      total += segment;

      ordered.push({
        ...next,
        routeSequence: ordered.length + 1,
        clusterId: `AREA-${String(clusterNo).padStart(2, '0')}`,
        clusterSize: cluster.length,
        routeReason:
          ordered.length === 0 && !hasStart
            ? 'START_POINT'
            : first
            ? tie
              ? 'AREA_TIE_PRIORITY'
              : 'NEAREST_AREA'
            : 'CLUSTER_PRIORITY',
        segmentDistanceKm: segment,
        routeSource: 'frontend',
      });

      first = false;
      prev = { lat: next.lat, lng: next.lng };
      pos = prev;
    }
  }

  return { ordered, total };
};

const planLocalRoute = (targets, start) => {
  if (!targets.length) return [];

  // Vessel start exists -> begin there
  if (start) return runLocalRoute(targets, start, true).ordered;

  // No vessel start -> try candidate starts, keep the shortest route
  const cLat = targets.reduce((s, t) => s + t.lat, 0) / targets.length;
  const cLng = targets.reduce((s, t) => s + t.lng, 0) / targets.length;

  const candidates = [...targets]
    .sort(
      (a, b) =>
        haversineDistanceKm(cLat, cLng, b.lat, b.lng) -
        haversineDistanceKm(cLat, cLng, a.lat, a.lng)
    )
    .slice(0, MAX_START_CANDIDATES);

  let best = null;
  candidates.forEach((c) => {
    const res = runLocalRoute(targets, { lat: c.lat, lng: c.lng }, false);
    const rank = riskRank(res.ordered[0].riskLevel);
    if (
      !best ||
      res.total < best.total - 1e-9 ||
      (Math.abs(res.total - best.total) <= 1e-9 && rank < best.rank)
    ) {
      best = { ...res, rank };
    }
  });

  return best.ordered;
};

// Subtitle line shown under each stop in the Detection Order list
const describeRouteStop = (target) => {
  if (!target.hasValidCoordinates) return 'No coordinates';

  const area = target.clusterId ?? 'Route point';
  const leg =
    target.segmentDistanceKm != null
      ? ` · ${Number(target.segmentDistanceKm).toFixed(2)} km leg`
      : '';
  const reason =
    target.routeReason === 'START_POINT'
      ? ' · start point'
      : target.routeReason === 'CLUSTER_PRIORITY'
      ? ' · priority in area'
      : target.routeReason === 'AREA_TIE_PRIORITY'
      ? ' · priority tie-break'
      : target.routeReason === 'NEAREST_AREA'
      ? ' · nearest area'
      : '';

  return `${area}${leg}${reason}`;
};

// ─────────────────────────────────────────────────────────────────────────────
// Map Resize
// ─────────────────────────────────────────────────────────────────────────────
function MapResizeHandler() {
  const map = useMap();

  useEffect(() => {
    const handleResize = () => {
      map.invalidateSize();
    };

    handleResize();

    const timer = setTimeout(handleResize, 350);

    return () => clearTimeout(timer);
  }, [map]);

  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Map Fitter
// ─────────────────────────────────────────────────────────────────────────────
function MapFitter({ points }) {
  const map = useMap();

  useEffect(() => {
    if (!points || points.length === 0) return;

    if (points.length === 1) {
      map.setView(points[0], Math.max(map.getZoom(), 13));
      return;
    }

    try {
      map.fitBounds(points, {
        padding: [50, 50],
        maxZoom: 15,
      });
    } catch {
      // Ignore invalid map geometry
    }
  }, [points, map]);

  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Route Direction Arrow (rotated to point from one leg's start toward its end,
// placed close to the next marker along the leg)
// ─────────────────────────────────────────────────────────────────────────────
const createDirectionArrowIcon = (bearing = 0) => {
  // The svg's arrow points east (0°) by default. Compass bearing 0° = north,
  // so rotate by (bearing - 90) to align the glyph with the travel direction.
  const rotation = bearing - 90;

  return L.divIcon({
    className: 'cleanup-route-arrow',
    html: `
      <div style="
        width: 26px;
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(13, 148, 136, 0.14);
        border: 2px solid #0d9488;
        border-radius: 50%;
        box-shadow: 0 3px 8px rgba(0,0,0,0.20);
        transform: rotate(${rotation}deg);
        transform-origin: center center;
      ">
        <svg
          width="15"
          height="15"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#0d9488"
          stroke-width="3"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <line x1="4" y1="12" x2="18" y2="12"></line>
          <polyline points="12 6 18 12 12 18"></polyline>
        </svg>
      </div>
    `,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// Numbered / Start (S) / End (E) Detection Marker
// ─────────────────────────────────────────────────────────────────────────────
const createNumberedIcon = (label, riskLevel, isSelected) => {
  const isHigh = String(riskLevel).toLowerCase() === 'high';
  const isMed = String(riskLevel).toLowerCase() === 'medium';

  const bgColor = isHigh
    ? '#e11d48'
    : isMed
    ? '#f59e0b'
    : '#10b981';

  const borderColor = isSelected ? '#0d9488' : '#ffffff';

  const ring = isSelected
    ? 'box-shadow: 0 0 0 3.5px rgba(13, 148, 136, 0.5), 0 6px 16px rgba(0,0,0,0.35); transform: scale(1.15);'
    : 'box-shadow: 0 3px 10px rgba(0,0,0,0.3);';

  return L.divIcon({
    className: 'custom-priority-marker',
    html: `
      <div style="
        background: ${bgColor};
        color: #ffffff;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'IBM Plex Mono', monospace, sans-serif;
        font-size: 12px;
        font-weight: 800;
        border: 2.5px solid ${borderColor};
        ${ring}
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: pointer;
      ">
        ${label}
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -18],
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// Vessel Start Marker
// ─────────────────────────────────────────────────────────────────────────────
const createStartIcon = () => {
  return L.divIcon({
    className: 'cleanup-vessel-start-marker',
    html: `
      <div style="
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #0f172a;
        color: #ffffff;
        border: 3px solid #ffffff;
        box-shadow: 0 0 0 3px rgba(15,23,42,0.25), 0 5px 14px rgba(0,0,0,0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: Arial, sans-serif;
        font-size: 12px;
        font-weight: 800;
      ">
        S
      </div>
    `,
    iconSize: [36, 36],
    iconAnchor: [18, 18],
    popupAnchor: [0, -20],
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export const CleanupPage = ({
  onNavigate,
  surveys = [],
  detections = [],
  activeSurvey = null,
  routePlan = null,
}) => {
  const { showToast } = useToast();

  // Selected survey
  const [selectedSurveyId, setSelectedSurveyId] = useState(
    activeSurvey?.id ? String(activeSurvey.id) : ''
  );

  const [filterMode, setFilterMode] = useState('all');
  const [selectedTargetId, setSelectedTargetId] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // ───────────────────────────────────────────────────────────────────────────
  // Keep selected survey synchronized with active survey
  // ───────────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (activeSurvey?.id != null) {
      setSelectedSurveyId(String(activeSurvey.id));
    }
  }, [activeSurvey]);

  // ───────────────────────────────────────────────────────────────────────────
  // Selected Survey
  // ───────────────────────────────────────────────────────────────────────────
  const selectedSurvey = useMemo(() => {
    if (selectedSurveyId) {
      return (
        surveys.find(
          (survey) => String(survey.id) === String(selectedSurveyId)
        ) || null
      );
    }

    return activeSurvey || surveys[0] || null;
  }, [surveys, selectedSurveyId, activeSurvey]);

  // ───────────────────────────────────────────────────────────────────────────
  // Survey ID
  // ───────────────────────────────────────────────────────────────────────────
  const selectedSurveyCode = selectedSurvey
    ? `SURV-${String(selectedSurvey.id).padStart(3, '0')}`
    : '—';

  // ───────────────────────────────────────────────────────────────────────────
  // Filter detections for ONLY selected survey
  // ───────────────────────────────────────────────────────────────────────────
  const surveyDetections = useMemo(() => {
    if (!selectedSurvey) return [];

    const surveyId = String(selectedSurvey.id);

    return detections.filter((d) => {
      const detectionSurveyId =
        d.survey_id ??
        d.surveyId ??
        d.survey_code ??
        d.surveyCode;

      if (detectionSurveyId == null) return false;

      const normalizedDetectionSurveyId = String(detectionSurveyId)
        .replace('SURV-', '')
        .trim();

      const normalizedSurveyId = surveyId
        .replace('SURV-', '')
        .trim();

      return normalizedDetectionSurveyId === normalizedSurveyId;
    });
  }, [detections, selectedSurvey]);

  // ───────────────────────────────────────────────────────────────────────────
  // Prepare detections
  // ───────────────────────────────────────────────────────────────────────────
  const allPrioritizedTargets = useMemo(() => {
    const mapped = surveyDetections
      .map((d, index) => {
        // Risk score
        let riskScore = Number(
          d.riskScore ??
            d.risk_score ??
            d.risk ??
            0
        );

        if (isNaN(riskScore)) {
          riskScore = 0;
        }

        if (riskScore > 0 && riskScore <= 1) {
          riskScore = riskScore * 100;
        }

        // Risk level
        let riskLevel = String(
          d.risk_level ??
            d.riskLevel ??
            d.risk_status ??
            ''
        ).toLowerCase();

        if (!['high', 'medium', 'low'].includes(riskLevel)) {
          if (riskScore >= 70) {
            riskLevel = 'high';
          } else if (riskScore >= 40) {
            riskLevel = 'medium';
          } else {
            riskLevel = 'low';
          }
        }

        // Detection ID
        const detectionId =
          d.detection_id ||
          d.detectionId ||
          (d.id
            ? String(d.id).startsWith('DET')
              ? d.id
              : `DET-${String(d.id).padStart(3, '0')}`
            : `DET-${String(index + 1).padStart(3, '0')}`);

        // Coordinates
        const latValue =
          d.lat ??
          d.latitude ??
          d.detection_latitude ??
          null;

        const lngValue =
          d.lng ??
          d.longitude ??
          d.detection_longitude ??
          null;

        const lat = Number(latValue);
        const lng = Number(lngValue);

        // Do NOT create fake coordinates
        const hasValidCoordinates =
          latValue != null &&
          lngValue != null &&
          !isNaN(lat) &&
          !isNaN(lng) &&
          !(lat === 0 && lng === 0);

        // Object class
        const objectClass =
          d.objectClass ||
          d.object_class ||
          d.class_name ||
          d.class ||
          'Marine Target';

        // Confidence
        const confValue = Number(d.confidence ?? 0);

        const confidence =
          confValue <= 1
            ? +(confValue * 100).toFixed(1)
            : +confValue.toFixed(1);

        // Depth
        const depthValue =
          d.depthMeters ??
          d.depth_meters ??
          d.depth ??
          selectedSurvey?.depth ??
          null;

        const depth =
          depthValue != null && !isNaN(Number(depthValue))
            ? Number(depthValue).toFixed(1)
            : '—';

        // Status
        const status =
          d.status ||
          d.cleanup_status ||
          'confirmed';

        return {
          key: `${selectedSurveyCode}-${detectionId}-${index}`,
          id: d.id,
          detectionId,
          surveyId: selectedSurveyCode,
          objectClass,
          confidence,
          riskScore: Math.round(riskScore),
          riskLevel,
          lat: hasValidCoordinates ? lat : null,
          lng: hasValidCoordinates ? lng : null,
          depth,
          status,
          hasValidCoordinates,
          raw: d,
        };
      })
      // Highest risk first
      .sort((a, b) => {
        if (b.riskScore !== a.riskScore) {
          return b.riskScore - a.riskScore;
        }

        return b.confidence - a.confidence;
      });

    return mapped.map((item, index) => ({
      ...item,
      priority: index + 1,
    }));
  }, [surveyDetections, selectedSurvey, selectedSurveyCode]);

  // ───────────────────────────────────────────────────────────────────────────
  // Display Targets
  // ───────────────────────────────────────────────────────────────────────────
  const displayTargets = useMemo(() => {
    if (filterMode === 'top5') {
      return allPrioritizedTargets.slice(0, 5);
    }

    return allPrioritizedTargets;
  }, [allPrioritizedTargets, filterMode]);

  // ───────────────────────────────────────────────────────────────────────────
  // Backend Route + Vessel Start
  // The Python route planner is the source of truth when routePlan is supplied.
  // No coordinates are invented here.
  // ───────────────────────────────────────────────────────────────────────────
  const normalizedRoutePlan = useMemo(() => {
    // Accept the direct Python response as well as common API wrappers.
    return (
      routePlan?.data?.route
        ? routePlan.data
        : routePlan?.mission_plan?.route
        ? routePlan.mission_plan
        : routePlan || null
    );
  }, [routePlan]);

  const backendRoute = useMemo(() => {
    const route = normalizedRoutePlan?.route;
    return Array.isArray(route) ? route : [];
  }, [normalizedRoutePlan]);

  const vesselStart = useMemo(() => {
    const start =
      normalizedRoutePlan?.vessel_start ||
      normalizedRoutePlan?.vesselStart ||
      selectedSurvey?.vessel_start ||
      selectedSurvey?.vesselStart ||
      selectedSurvey?.start_position ||
      selectedSurvey?.startPosition ||
      null;

    if (!start) return null;

    const lat = Number(start.latitude ?? start.lat);
    const lng = Number(start.longitude ?? start.lng);

    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
    if (lat === 0 && lng === 0) return null;

    return { lat, lng };
  }, [normalizedRoutePlan, selectedSurvey]);

  // ───────────────────────────────────────────────────────────────────────────
  // Route Targets (raw order, before S / E labels)
  // Follow Python backend sequence exactly when routePlan.route exists.
  // Otherwise fall back to the local planner (vessel start, or best auto start).
  // ───────────────────────────────────────────────────────────────────────────
  const rawRouteTargets = useMemo(() => {
    // If Python supplied a route, use its coordinates and sequence directly.
    // Do not require a second frontend survey/detection join for map rendering.
    if (backendRoute.length > 0) {
      const targetLookup = new Map(
        allPrioritizedTargets.map((target) => [String(target.detectionId), target])
      );

      return backendRoute
        .slice()
        .sort(
          (a, b) =>
            Number(a.sequence ?? a.route_sequence ?? a.route_order ?? 0) -
            Number(b.sequence ?? b.route_sequence ?? b.route_order ?? 0)
        )
        .map((stop, index) => {
          const detectionId =
            stop.detection_id ?? stop.detectionId ?? stop.id ?? `ROUTE-${index + 1}`;

          const lat = Number(stop.latitude ?? stop.lat);
          const lng = Number(stop.longitude ?? stop.lng);
          const existing = targetLookup.get(String(detectionId));

          if (!Number.isFinite(lat) || !Number.isFinite(lng) || (lat === 0 && lng === 0)) {
            return null;
          }

          const riskScore = Number(
            stop.risk_score ?? stop.riskScore ?? existing?.riskScore ?? 0
          );
          const riskLevel = String(
            stop.risk_level ?? stop.riskLevel ?? existing?.riskLevel ??
              (riskScore >= 70 ? 'high' : riskScore >= 40 ? 'medium' : 'low')
          ).toLowerCase();

          return {
            ...(existing || {}),
            key: existing?.key || `route-${detectionId}-${index}`,
            detectionId,
            objectClass: stop.object_class ?? stop.objectClass ?? existing?.objectClass ?? 'Marine Target',
            confidence: Number(stop.confidence ?? existing?.confidence ?? 0),
            riskScore: Math.round(riskScore),
            riskLevel,
            priority: existing?.priority ?? index + 1,
            priorityScore: Number(stop.priority_score ?? stop.priorityScore ?? existing?.priorityScore ?? 0),
            priorityRank: stop.priority_rank ?? existing?.priorityRank ?? null,
            clusterSize: stop.cluster_size ?? null,
            routeScore: Number(stop.route_score ?? stop.routeScore ?? 0),
            distanceFromVesselKm: Number(stop.distance_from_vessel_km ?? 0),
            segmentDistanceKm: Number(stop.segment_distance_km ?? 0),
            routeSequence: Number(stop.sequence ?? stop.route_sequence ?? stop.route_order ?? index + 1),
            clusterId: stop.cluster_id ?? existing?.clusterId ?? null,
            routeReason: stop.route_reason ?? existing?.routeReason ?? null,
            lat,
            lng,
            hasValidCoordinates: true,
            depth: stop.depth ?? stop.depth_meters ?? existing?.depth ?? '—',
            status: stop.status ?? stop.cleanup_status ?? existing?.status ?? 'confirmed',
            routeSource: 'backend',
          };
        })
        .filter(Boolean);
    }

    // Fallback: vessel start if present, otherwise the best auto-selected start
    return planLocalRoute(
      displayTargets.filter((t) => t.hasValidCoordinates),
      vesselStart
    );
  }, [allPrioritizedTargets, displayTargets, backendRoute, vesselStart]);

  // ───────────────────────────────────────────────────────────────────────────
  // Route Targets with display labels
  //   S = first stop (only when there is no vessel start; otherwise the vessel
  //       marker itself is S), E = last stop, others keep their number.
  // ───────────────────────────────────────────────────────────────────────────
  const routeTargets = useMemo(() => {
    const last = rawRouteTargets.length - 1;

    return rawRouteTargets.map((t, i) => ({
      ...t,
      routeLabel:
        i === 0 && !vesselStart
          ? 'S'
          : i === last
          ? 'E'
          : t.routeSequence ?? i + 1,
    }));
  }, [rawRouteTargets, vesselStart]);

  // ───────────────────────────────────────────────────────────────────────────
  // Ordered Targets for the UI
  // Route order first, then targets that have no coordinates
  // ───────────────────────────────────────────────────────────────────────────
  const orderedTargets = useMemo(() => {
    const noCoords = displayTargets.filter((t) => !t.hasValidCoordinates);
    return [...routeTargets, ...noCoords];
  }, [routeTargets, displayTargets]);

  // ───────────────────────────────────────────────────────────────────────────
  // Selected Target
  // ───────────────────────────────────────────────────────────────────────────
  const selectedTarget = useMemo(() => {
    if (!selectedTargetId) return orderedTargets[0] || null;

    return (
      orderedTargets.find(
        (t) => t.detectionId === selectedTargetId || t.key === selectedTargetId
      ) ||
      orderedTargets[0] ||
      null
    );
  }, [orderedTargets, selectedTargetId]);

  // ───────────────────────────────────────────────────────────────────────────
  // Route Points
  // Vessel start is the first point (if any), followed by route stops.
  // ───────────────────────────────────────────────────────────────────────────
  const routePolylinePoints = useMemo(() => {
    const points = [];

    if (vesselStart) {
      points.push([vesselStart.lat, vesselStart.lng]);
    }

    routeTargets.forEach((target) => {
      points.push([target.lat, target.lng]);
    });

    return points;
  }, [routeTargets, vesselStart]);

  // ───────────────────────────────────────────────────────────────────────────
  // Route Distance
  // ───────────────────────────────────────────────────────────────────────────
  const routeMetrics = useMemo(() => {
    if (routeTargets.length === 0) {
      return { totalDistanceKm: 0, legDistances: [] };
    }

    let totalKm = 0;
    const legDistances = [];

    if (vesselStart) {
      const first = routeTargets[0];
      const distance = haversineDistanceKm(
        vesselStart.lat,
        vesselStart.lng,
        first.lat,
        first.lng
      );
      legDistances.push(distance);
      totalKm += distance;
    }

    for (let i = 1; i < routeTargets.length; i++) {
      const previous = routeTargets[i - 1];
      const current = routeTargets[i];
      const distance = haversineDistanceKm(
        previous.lat,
        previous.lng,
        current.lat,
        current.lng
      );
      legDistances.push(distance);
      totalKm += distance;
    }

    return {
      totalDistanceKm: Number(totalKm.toFixed(2)),
      legDistances,
    };
  }, [routeTargets, vesselStart]);

  // ───────────────────────────────────────────────────────────────────────────
  // Route Direction Arrows — one per leg, placed close to the next marker,
  // rotated to face the travel direction of that leg
  // ───────────────────────────────────────────────────────────────────────────
  const routeDirectionArrows = useMemo(() => {
    const points = routePolylinePoints;
    if (points.length < 2) return [];

    const NEAR_NEXT_FRACTION = 0.72;
    const arrows = [];

    for (let i = 0; i < points.length - 1; i++) {
      const from = points[i];
      const to = points[i + 1];
      const bearing = calculateBearing(from[0], from[1], to[0], to[1]);
      const arrowLat = from[0] + (to[0] - from[0]) * NEAR_NEXT_FRACTION;
      const arrowLng = from[1] + (to[1] - from[1]) * NEAR_NEXT_FRACTION;

      arrows.push({
        key: `arrow-${i}-${from[0]}-${from[1]}-${to[0]}-${to[1]}`,
        position: [arrowLat, arrowLng],
        bearing,
      });
    }

    return arrows;
  }, [routePolylinePoints]);

  // ───────────────────────────────────────────────────────────────────────────
  // Statistics
  // ───────────────────────────────────────────────────────────────────────────
  const stats = useMemo(() => {
    const total = displayTargets.length;

    const high = displayTargets.filter(
      (target) => target.riskLevel === 'high'
    ).length;

    const medium = displayTargets.filter(
      (target) => target.riskLevel === 'medium'
    ).length;

    const low = displayTargets.filter(
      (target) => target.riskLevel === 'low'
    ).length;

    return {
      total,
      high,
      medium,
      low,
    };
  }, [displayTargets]);

  // ───────────────────────────────────────────────────────────────────────────
  // Select Target
  // ───────────────────────────────────────────────────────────────────────────
  const handleTargetSelect = (target) => {
    setSelectedTargetId(target.detectionId);
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Survey Change
  // ───────────────────────────────────────────────────────────────────────────
  const handleSurveyChange = (e) => {
    const value = e.target.value;

    setSelectedSurveyId(value);
    setSelectedTargetId(null);

    showToast({
      type: 'info',
      message: `Showing cleanup targets for SURV-${String(value).padStart(
        3,
        '0'
      )}.`,
    });
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Generate Route
  // ───────────────────────────────────────────────────────────────────────────
  const handleGenerateRoute = () => {
    if (!selectedSurvey) {
      showToast({
        type: 'warning',
        message: 'Select a survey first.',
      });
      return;
    }

    if (routeTargets.length === 0) {
      showToast({
        type: 'warning',
        message: 'No detections with valid coordinates are available for this survey.',
      });
      return;
    }

    setIsGenerating(true);

    setTimeout(() => {
      setIsGenerating(false);

      showToast({
        type: 'success',
        message: `Cleanup route generated for ${routeTargets.length} targets.`,
      });
    }, 400);
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Export Mission Plan
  // ───────────────────────────────────────────────────────────────────────────
  const handleExportMissionPlan = () => {
    if (!selectedSurvey) {
      showToast({
        type: 'warning',
        message: 'Select a survey first.',
      });
      return;
    }

    const missionPlan = {
      surveyId: selectedSurveyCode,
      surveyName: selectedSurvey.name || 'Cleanup Survey',
      vesselName: selectedSurvey.vessel || null,
      waterBody:
        selectedSurvey.water_body ||
        selectedSurvey.location ||
        null,
      totalTargets: displayTargets.length,
      routeTargets: routeTargets.length,
      totalDistanceKm: normalizedRoutePlan?.estimated_total_distance_km ?? routeMetrics.totalDistanceKm,
      vesselStart,
      routeSource: backendRoute.length > 0 ? 'Python backend route planner' : 'Frontend fallback',

      waypoints: routeTargets.map((target, index) => ({
        order: target.routeSequence ?? index + 1,
        label: target.routeLabel,
        priority: target.priority,
        priorityScore: target.priorityScore ?? null,
        priorityRank: target.priorityRank ?? null,
        routeScore: target.routeScore ?? null,
        distanceFromVesselKm: target.distanceFromVesselKm ?? null,
        segmentDistanceKm: target.segmentDistanceKm ?? null,
        clusterId: target.clusterId ?? null,
        routeReason: target.routeReason ?? null,
        detectionId: target.detectionId,
        objectClass: target.objectClass,
        confidence: target.confidence,
        riskScore: target.riskScore,
        riskLevel: target.riskLevel.toUpperCase(),
        latitude: target.lat,
        longitude: target.lng,
        depthMeters: target.depth,
        status: target.status,
      })),
    };

    const blob = new Blob(
      [JSON.stringify(missionPlan, null, 2)],
      {
        type: 'application/json',
      }
    );

    const url = URL.createObjectURL(blob);

    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `AquaVision_${selectedSurveyCode}_Cleanup_Route.json`;

    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);

    URL.revokeObjectURL(url);

    showToast({
      type: 'success',
      message: 'Cleanup route exported successfully.',
    });
  };

  // ───────────────────────────────────────────────────────────────────────────
  // No Survey
  // ───────────────────────────────────────────────────────────────────────────
  if (!selectedSurvey) {
    return (
      <section className="animate-page-fade space-y-4 sm:space-y-5 bg-[#EFF8FB] min-h-screen -m-3 sm:-m-6 p-3 sm:p-6 overflow-x-hidden">
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <h2 className="text-[16px] font-bold text-slate-900 mb-2">
            Cleanup Priority
          </h2>

          <p className="text-[12px] text-slate-600">
            Create a survey and add detections before planning cleanup.
          </p>

          <button
            onClick={() => onNavigate?.('surveys')}
            className="mt-4 bg-teal-600 hover:bg-teal-700 text-white text-[12px] font-semibold px-4 py-2 rounded-lg"
          >
            Go to Surveys
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="animate-page-fade space-y-4 sm:space-y-5 bg-[#EFF8FB] min-h-screen -m-3 sm:-m-6 p-3 sm:p-6 overflow-x-hidden">

      {/* ─────────────────────────────────────────────────────────────────────
          HEADER
      ───────────────────────────────────────────────────────────────────── */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 pb-3 border-b border-slate-200">

        <div>
          <h1 className="text-[16px] font-bold text-slate-900">
            Cleanup Priority
          </h1>

          <p className="text-[12px] text-slate-600 mt-1">
            Review high-risk targets and plan the cleanup route.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 lg:flex lg:flex-wrap items-stretch gap-2 w-full lg:w-auto">

          {/* Survey Filter */}
          <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 shadow-sm">

            <span className="text-[10.5px] font-bold uppercase tracking-wide text-slate-500">
              Survey :  
            </span>

            <select
              value={String(selectedSurvey.id ?? '')}
              onChange={handleSurveyChange}
              className="
                h-8
                min-w-0
                w-full
                sm:max-w-[210px]
                bg-white
                border-0
                text-[11.5px]
                font-mono
                font-bold
                text-slate-800
                focus:outline-none
                focus:ring-0
                cursor-pointer
              "
            >
              {surveys.map((survey) => (
                <option
                  key={survey.id ?? survey.name ?? 'survey-option'}
                  value={survey.id ?? ''}
                >
                  SURV-{String(survey.id).padStart(3, '0')} ·{' '}
                  {survey.name || 'Survey'}
                </option>
              ))}
            </select>

          </div>

          {/* Top 5 / All */}
          <div className="bg-white border border-slate-200 rounded-lg p-1 flex items-center justify-center shadow-sm w-full sm:w-auto">

            <button
              onClick={() => setFilterMode('top5')}
              className={`px-3 py-1.5 text-[11.5px] font-bold rounded-md transition-all ${
                filterMode === 'top5'
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
              }`}
            >
              Top 5
            </button>

            <button
              onClick={() => setFilterMode('all')}
              className={`px-3 py-1.5 text-[11.5px] font-bold rounded-md transition-all ${
                filterMode === 'all'
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
              }`}
            >
              All ({allPrioritizedTargets.length})
            </button>

          </div>

          {/* Generate */}
          <button
            onClick={handleGenerateRoute}
            disabled={isGenerating}
            className="
              bg-teal-600
              hover:bg-teal-700
              disabled:opacity-50
              text-white
              font-bold
              text-[12px]
              py-2
              px-4
              rounded-lg
              flex
              justify-center
              w-full sm:w-auto
              items-center
              gap-2
              shadow-sm
              transition-colors
              whitespace-nowrap
            "
          >
            {isGenerating ? (
              <span className="w-3.5 h-3.5 border-2 border-white/40 border-t-white rounded-full animate-spin" />
            ) : (
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                className="w-4 h-4"
              >
                <polygon points="3 11 22 2 13 21 11 13 3 11" />
              </svg>
            )}

            <span>Generate Route</span>
          </button>
        </div>
      </div>


      {/* ─────────────────────────────────────────────────────────────────────
          SELECTED SURVEY INFO — with stat cards
      ───────────────────────────────────────────────────────────────────── */}
      <div className="bg-white border border-teal-200 rounded-xl px-4 py-3 shadow-sm">

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">

          <div className="flex items-center gap-3">

            <span className="w-2.5 h-2.5 rounded-full bg-teal-500 animate-pulse" />

            <div>
              <div className="flex items-center gap-2">

                <span className="font-mono text-[11px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded">
                  {selectedSurveyCode}
                </span>

                <span className="text-[13px] font-bold text-slate-900">
                  {selectedSurvey.name || 'Survey'}
                </span>

              </div>

              <p className="text-[10.5px] text-slate-500 mt-1">
                Cleanup targets from this survey only
              </p>
            </div>

          </div>

          {/* Stat cards */}
          <div className="grid grid-cols-3 gap-2 w-full sm:w-auto">

            <div className="bg-slate-50 border border-slate-200 rounded-lg px-3.5 py-2 text-center min-w-0">
              <div className="text-[9px] uppercase tracking-wide text-slate-500 font-bold">
                Targets
              </div>
              <div className="text-[16px] font-mono font-extrabold text-slate-900 mt-0.5 leading-none">
                {stats.total}
              </div>
            </div>

            <div className="bg-rose-50 border border-rose-200 rounded-lg px-3.5 py-2 text-center min-w-0">
              <div className="text-[9px] uppercase tracking-wide text-rose-600 font-bold">
                High Risk
              </div>
              <div className="text-[16px] font-mono font-extrabold text-rose-700 mt-0.5 leading-none">
                {stats.high}
              </div>
            </div>

            <div className="bg-teal-50 border border-teal-200 rounded-lg px-3.5 py-2 text-center min-w-0">
              <div className="text-[9px] uppercase tracking-wide text-teal-600 font-bold">
                Route Points
              </div>
              <div className="text-[16px] font-mono font-extrabold text-teal-700 mt-0.5 leading-none">
                {routeTargets.length}
              </div>
            </div>

          </div>

        </div>

      </div>

      {/* ─────────────────────────────────────────────────────────────────────
          MAIN CONTENT
      ───────────────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_340px] gap-4 sm:gap-5 items-start min-w-0">

        {/* LEFT */}
        <div className="space-y-4 min-w-0">

          {/* MAP */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">

            <div className="px-4 py-3 bg-slate-50/80 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">

              <div className="flex items-center gap-2">

                <span className="w-2.5 h-2.5 rounded-full bg-teal-500" />

                <span className="text-[13px] font-bold text-slate-800">
                  Cleanup Route
                </span>

                <span className="text-[10.5px] font-mono font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded">
                  {routeTargets.length} Points
                </span>

                {backendRoute.length > 0 && (
                  <span className="text-[10px] font-mono font-bold text-indigo-700 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded">
                    Python Route
                  </span>
                )}

              </div>

              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10.5px] font-medium text-slate-600">

                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-600" />
                  High
                </span>

                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                  Medium
                </span>

                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                  Low
                </span>

              </div>

            </div>

            <div className="relative h-[390px] sm:h-[470px] lg:h-[530px] w-full">

              {routeTargets.length === 0 ? (

                <div className="absolute inset-0 flex items-center justify-center bg-slate-50">

                  <div className="text-center px-6">

                    <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-teal-50 border border-teal-200 flex items-center justify-center">

                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        className="w-6 h-6 text-teal-600"
                      >
                        <path d="M12 21s7-6.2 7-12a7 7 0 1 0-14 0c0 5.8 7 12 7 12Z" />
                        <circle cx="12" cy="9" r="2.5" />
                      </svg>

                    </div>

                    <h3 className="text-[13px] font-bold text-slate-800">
                      No route points available
                    </h3>

                    <p className="text-[11px] text-slate-500 mt-1">
                      This survey has no detections with valid coordinates.
                    </p>

                  </div>

                </div>

              ) : (

                <MapContainer
                  center={routePolylinePoints[0]}
                  zoom={13}
                  style={{
                    height: '100%',
                    width: '100%',
                  }}
                  className="z-0"
                >

                  <MapResizeHandler />

                  <MapFitter
                    points={routePolylinePoints}
                  />

                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    maxZoom={19}
                  />

                  {/* Route glow */}
                  {routePolylinePoints.length > 1 && (
                    <Polyline
                      positions={routePolylinePoints}
                      pathOptions={{
                        color: '#2dd4bf',
                        weight: 8,
                        opacity: 0.25,
                        lineCap: 'round',
                      }}
                    />
                  )}

                  {/* Main route */}
                  {routePolylinePoints.length > 1 && (
                    <Polyline
                      positions={routePolylinePoints}
                      pathOptions={{
                        color: '#0d9488',
                        weight: 4,
                        opacity: 0.95,
                        lineCap: 'round',
                        lineJoin: 'round',
                      }}
                    />
                  )}

                  {/* Vessel start marker */}
                  {vesselStart && (
                    <Marker
                      position={[vesselStart.lat, vesselStart.lng]}
                      icon={createStartIcon()}
                    >
                      <Popup>
                        <div className="p-1 font-sans text-slate-800 min-w-[180px]">
                          <div className="font-bold text-slate-900">Vessel Start</div>
                          <div className="text-[11px] text-slate-500 mt-1">
                            Mission route origin
                          </div>
                          <div className="font-mono text-[11px] mt-2">
                            {vesselStart.lat.toFixed(5)}°, {vesselStart.lng.toFixed(5)}°
                          </div>
                        </div>
                      </Popup>
                    </Marker>
                  )}

                  {/* Route stops: S (no vessel), 1, 2, 3 ... E */}
                  {routeTargets.map((target, idx) => {
                    const isSelected =
                      selectedTarget?.detectionId === target.detectionId;

                    const markerLabel =
                      target.routeLabel ?? target.routeSequence ?? idx + 1;

                    return (
                      <Marker
                        key={target.key}
                        position={[target.lat, target.lng]}
                        icon={createNumberedIcon(
                          markerLabel,
                          target.riskLevel,
                          isSelected
                        )}
                        eventHandlers={{
                          click: () => handleTargetSelect(target),
                        }}
                      >
                        <Popup>
                          <div className="p-1 font-sans text-slate-800 space-y-2 min-w-[220px]">
                            <div className="flex items-center justify-between pb-1.5 border-b border-slate-200">
                              <span className="font-mono text-[11px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded">
                                Route #{markerLabel}
                              </span>

                              <span
                                className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                                  target.riskLevel === 'high'
                                    ? 'bg-rose-100 text-rose-800'
                                    : target.riskLevel === 'medium'
                                    ? 'bg-amber-100 text-amber-800'
                                    : 'bg-emerald-100 text-emerald-800'
                                }`}
                              >
                                {target.riskLevel}
                              </span>
                            </div>

                            <div className="space-y-1 text-[11px]">
                              <div className="flex justify-between gap-4">
                                <span className="text-slate-500">Detection</span>
                                <span className="font-mono font-bold">{target.detectionId}</span>
                              </div>

                              <div className="flex justify-between gap-4">
                                <span className="text-slate-500">Object</span>
                                <span className="font-semibold">{target.objectClass}</span>
                              </div>

                              <div className="flex justify-between gap-4">
                                <span className="text-slate-500">Confidence</span>
                                <span className="font-mono font-bold">{target.confidence}%</span>
                              </div>

                              <div className="flex justify-between gap-4">
                                <span className="text-slate-500">Risk</span>
                                <span className="font-mono font-bold">{target.riskScore}/100</span>
                              </div>

                              {target.priorityScore != null && (
                                <div className="flex justify-between gap-4">
                                  <span className="text-slate-500">Priority Score</span>
                                  <span className="font-mono font-bold">{Number(target.priorityScore).toFixed(2)}</span>
                                </div>
                              )}

                              {target.routeScore != null && target.routeScore !== 0 && (
                                <div className="flex justify-between gap-4">
                                  <span className="text-slate-500">Route Score</span>
                                  <span className="font-mono font-bold">{Number(target.routeScore).toFixed(2)}</span>
                                </div>
                              )}

                              <div className="flex justify-between gap-4">
                                <span className="text-slate-500">Latitude</span>
                                <span className="font-mono font-bold">{target.lat.toFixed(5)}°</span>
                              </div>

                              <div className="flex justify-between gap-4">
                                <span className="text-slate-500">Longitude</span>
                                <span className="font-mono font-bold">{target.lng.toFixed(5)}°</span>
                              </div>

                              <div className="flex justify-between gap-4">
                                <span className="text-slate-500">Depth</span>
                                <span className="font-mono font-bold">{target.depth} m</span>
                              </div>
                            </div>
                          </div>
                        </Popup>
                      </Marker>
                    );
                  })}

                  {/* Per-leg direction arrows — one for every route segment,
                      rotated to face the travel direction and placed close
                      to the next marker on that leg */}
                  {routeDirectionArrows.map((arrow) => (
                    <Marker
                      key={arrow.key}
                      position={arrow.position}
                      icon={createDirectionArrowIcon(arrow.bearing)}
                      interactive={false}
                    />
                  ))}

                </MapContainer>

              )}

            </div>
          </div>

          {/* ─────────────────────────────────────────────────────────────────
              DETECTION ORDER
          ───────────────────────────────────────────────────────────────── */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">

            <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-100">

              <div>
                <div className="text-[12.5px] font-bold text-slate-800">
                  Detection Order
                </div>

                <div className="text-[10.5px] text-slate-500 mt-0.5">
                  Nearest area first; priority decides order when areas are close or inside a cleanup area.
                </div>
              </div>

              <span className="font-mono text-[10.5px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-1 rounded">
                {displayTargets.length} Targets
              </span>

            </div>

            <div className="space-y-2 min-w-0">

              {orderedTargets.length === 0 ? (

                <div className="py-6 text-center text-[11px] text-slate-500">
                  No detections found for this survey.
                </div>

              ) : (

                <div className="space-y-2 min-w-0">
                  {orderedTargets.map((target) => {
                    const isSelected =
                      selectedTarget?.detectionId === target.detectionId;

                    const isHigh = target.riskLevel === 'high';
                    const isMedium = target.riskLevel === 'medium';

                    const badgeColor = isHigh
                      ? 'bg-rose-500'
                      : isMedium
                      ? 'bg-amber-500'
                      : 'bg-emerald-500';

                    const riskBadge = isHigh
                      ? 'bg-rose-100 text-rose-700'
                      : isMedium
                      ? 'bg-amber-100 text-amber-700'
                      : 'bg-emerald-100 text-emerald-700';

                    return (
                      <div
                        key={target.key}
                        className={`cursor-pointer rounded-lg border px-2.5 sm:px-3 py-2.5 transition-all ${
                          isSelected
                            ? 'border-teal-500 bg-teal-50/50 shadow-sm ring-1 ring-teal-400'
                            : 'border-slate-200 bg-slate-50/50 hover:bg-slate-100'
                        }`}
                        onClick={() => handleTargetSelect(target)}
                      >
                        {/* Desktop row */}
                        <div className="hidden sm:grid grid-cols-[34px_minmax(0,1fr)_74px_74px_70px] items-center gap-3">
                          <span
                            className={`w-7 h-7 rounded-full text-white font-mono text-[11px] font-bold flex items-center justify-center ${badgeColor}`}
                          >
                            {target.routeLabel ?? '–'}
                          </span>

                          <div className="min-w-0">
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="text-[12px] font-bold text-slate-900 truncate">
                                {target.objectClass}
                              </span>
                              <span className="font-mono text-[10px] text-slate-500 truncate max-w-[110px]">
                                {target.detectionId}
                              </span>
                            </div>
                            <div className="text-[10px] text-slate-500 mt-0.5">
                              {describeRouteStop(target)}
                            </div>
                          </div>

                          <div className="text-right">
                            <div className="text-[9.5px] uppercase tracking-wide text-slate-400 font-bold">
                              Risk Score
                            </div>
                            <div className="font-mono text-[12px] font-bold text-slate-800 leading-tight">
                              {target.riskScore}
                              <span className="text-[9px] text-slate-400 font-normal"> %</span>
                            </div>
                          </div>

                          <div className="text-right leading-tight">
                            <div className="text-[9.5px] uppercase tracking-wide text-slate-400 font-bold leading-none">
                              Risk Level
                            </div>
                            <span
                              className={`inline-block mt-0.5 text-[9px] font-bold uppercase px-1.5 py-0.5 rounded leading-none ${riskBadge}`}
                            >
                              {target.riskLevel}
                            </span>
                          </div>

                          <div className="text-right">
                            <div className="text-[9.5px] uppercase tracking-wide text-slate-400 font-bold">
                              Depth
                            </div>
                            <div className="font-mono text-[11px] font-bold text-slate-800">
                              {target.depth}
                              {target.depth !== '—' && ' m'}
                            </div>
                          </div>
                        </div>

                        {/* Mobile card */}
                        <div className="sm:hidden">
                          <div className="flex items-center gap-2 min-w-0">
                            <span
                              className={`shrink-0 w-7 h-7 rounded-full text-white font-mono text-[11px] font-bold flex items-center justify-center ${badgeColor}`}
                            >
                              {target.routeLabel ?? '–'}
                            </span>

                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-1.5 min-w-0">
                                <span className="text-[12px] font-bold text-slate-900 truncate">
                                  {target.objectClass}
                                </span>
                                <span className="font-mono text-[9.5px] text-slate-500 truncate max-w-[90px] shrink-0">
                                  {target.detectionId}
                                </span>
                              </div>
                              <div className="text-[9.5px] text-slate-500 mt-0.5">
                                {describeRouteStop(target)}
                              </div>
                            </div>

                            <span
                              className={`shrink-0 text-[9px] font-bold uppercase px-1.5 py-0.5 rounded ${riskBadge}`}
                            >
                              {target.riskLevel}
                            </span>
                          </div>

                          <div className="grid grid-cols-2 gap-2 mt-2.5 pt-2 border-t border-slate-200">
                            <div className="bg-white rounded-md border border-slate-200 px-2.5 py-1.5">
                              <div className="text-[8.5px] uppercase tracking-wide text-slate-400 font-bold">
                                Risk Score
                              </div>
                              <div className="font-mono text-[11px] font-bold text-slate-800">
                                {target.riskScore}%
                              </div>
                            </div>

                            <div className="bg-white rounded-md border border-slate-200 px-2.5 py-1.5">
                              <div className="text-[8.5px] uppercase tracking-wide text-slate-400 font-bold">
                                Depth
                              </div>
                              <div className="font-mono text-[11px] font-bold text-slate-800">
                                {target.depth}
                                {target.depth !== '—' && ' m'}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

            </div>

          </div>

        </div>

        {/* ───────────────────────────────────────────────────────────────────
            RIGHT COLUMN
        ─────────────────────────────────────────────────────────────────── */}
        <div className="space-y-4 min-w-0">

          {/* Route Summary */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4 md:p-5">

            <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-100">

              <div className="flex items-center gap-2">

                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  className="w-4 h-4 text-teal-600"
                >
                  <path d="M4 19L20 5" />
                  <circle cx="5" cy="19" r="2" />
                  <circle cx="19" cy="5" r="2" />
                </svg>

                <h2 className="text-[13.5px] font-bold text-slate-900">
                  Cleanup Summary
                </h2>

              </div>

              <span className="font-mono text-[10px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded">
                {selectedSurveyCode}
              </span>

            </div>

            <div className="space-y-2.5 text-[12px]">

              {/* Total */}
              <div className="flex justify-between items-center py-1.5 border-b border-dashed border-slate-200">

                <span className="text-slate-600">
                  Total detections
                </span>

                <span className="font-mono font-bold text-slate-900 bg-slate-100 px-2 py-0.5 rounded">
                  {stats.total}
                </span>

              </div>

              {/* High */}
              <div className="flex justify-between items-center py-1.5 border-b border-dashed border-slate-200">

                <span className="text-slate-600">
                  High risk
                </span>

                <span className="font-mono font-bold text-rose-700 bg-rose-50 border border-rose-200 px-2 py-0.5 rounded">
                  {stats.high}
                </span>

              </div>

              {/* Medium */}
              <div className="flex justify-between items-center py-1.5 border-b border-dashed border-slate-200">

                <span className="text-slate-600">
                  Medium risk
                </span>

                <span className="font-mono font-bold text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded">
                  {stats.medium}
                </span>

              </div>

              {/* Low */}
              <div className="flex justify-between items-center py-1.5 border-b border-dashed border-slate-200">

                <span className="text-slate-600">
                  Low risk
                </span>

                <span className="font-mono font-bold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded">
                  {stats.low}
                </span>

              </div>

              {/* Route points */}
              <div className="flex justify-between items-center py-1.5 border-b border-dashed border-slate-200">

                <span className="text-slate-600">
                  Route points
                </span>

                <span className="font-mono font-bold text-teal-800 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded">
                  {routeTargets.length}
                </span>

              </div>

              {/* Distance */}
              <div className="flex justify-between items-center py-1.5">

                <span className="text-slate-600">
                  Route distance
                </span>

                <span className="font-mono font-bold text-teal-800 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded">
                  {routeMetrics.totalDistanceKm} km
                </span>

              </div>

            </div>

            <div className="space-y-2 mt-4 pt-3 border-t border-slate-100">

              <button
                onClick={handleGenerateRoute}
                disabled={
                  isGenerating ||
                  routeTargets.length === 0
                }
                className="
                  bg-teal-600
                  hover:bg-teal-700
                  disabled:opacity-50
                  text-white
                  font-bold
                  text-[12px]
                  py-2
                  w-full
                  justify-center
                  rounded-lg
                  shadow-sm
                "
              >
                {isGenerating
                  ? 'Generating...'
                  : 'Generate Route'}
              </button>

              <button
                onClick={handleExportMissionPlan}
                className="
                  bg-white
                  hover:bg-slate-50
                  border
                  border-slate-200
                  text-slate-700
                  font-semibold
                  text-[12px]
                  py-2
                  w-full
                  justify-center
                  rounded-lg
                  shadow-sm
                "
              >
                Export Route
              </button>

              <button
                onClick={() => onNavigate?.('map')}
                className="
                  text-teal-700
                  hover:text-teal-800
                  hover:bg-teal-50
                  border
                  border-teal-200
                  text-[12px]
                  py-1.5
                  w-full
                  justify-center
                  rounded-lg
                  font-semibold
                "
              >
                View Risk Map →
              </button>

            </div>

          </div>

          {/* Selected Target */}
          {selectedTarget && (
            <div className="bg-white border border-teal-300 rounded-xl shadow-sm p-4 md:p-5">

              <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-100">

                <div className="flex items-center gap-2">

                  <span className="w-2.5 h-2.5 rounded-full bg-teal-600" />

                  <h3 className="text-[13px] font-bold text-slate-900">
                    Selected Detection
                  </h3>

                </div>

                <span className="font-mono text-[10.5px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded">
                  #{selectedTarget.routeLabel ?? selectedTarget.priority}
                </span>

              </div>

              <div className="space-y-2 text-[12px] min-w-0">

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">
                    Detection ID
                  </span>

                  <span className="font-mono font-bold text-slate-800">
                    {selectedTarget.detectionId}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">
                    Object
                  </span>

                  <span className="font-bold text-slate-900">
                    {selectedTarget.objectClass}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">
                    Confidence
                  </span>

                  <span className="font-mono font-bold text-teal-700">
                    {selectedTarget.confidence}%
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">
                    Risk
                  </span>

                  <span className="font-mono font-bold text-slate-900">
                    {selectedTarget.riskScore}/100
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">
                    Risk level
                  </span>

                  <span
                    className={`
                      text-[10px]
                      px-2
                      py-0.5
                      rounded
                      uppercase
                      font-bold
                      ${
                        selectedTarget.riskLevel === 'high'
                          ? 'bg-rose-100 text-rose-800'
                          : selectedTarget.riskLevel === 'medium'
                          ? 'bg-amber-100 text-amber-800'
                          : 'bg-emerald-100 text-emerald-800'
                      }
                    `}
                  >
                    {selectedTarget.riskLevel}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">
                    Latitude
                  </span>

                  <span className="font-mono font-bold text-slate-800">
                    {selectedTarget.hasValidCoordinates
                      ? `${selectedTarget.lat.toFixed(5)}° N`
                      : '—'}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-slate-100">
                  <span className="text-slate-500">
                    Longitude
                  </span>

                  <span className="font-mono font-bold text-slate-800">
                    {selectedTarget.hasValidCoordinates
                      ? `${selectedTarget.lng.toFixed(5)}° E`
                      : '—'}
                  </span>
                </div>

                <div className="flex justify-between py-1">
                  <span className="text-slate-500">
                    Depth
                  </span>

                  <span className="font-mono font-bold text-slate-800">
                    {selectedTarget.depth}
                    {selectedTarget.depth !== '—' && ' m'}
                  </span>
                </div>

              </div>

            </div>
          )}

        </div>
      </div>
    </section>
  );
};

export default CleanupPage;