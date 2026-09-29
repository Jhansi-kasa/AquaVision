/**
 * Aqua Vision — Data & Telemetry Structure
 * Empty detection state — all detection data is generated dynamically by the YOLO detection engine.
 */

export const activeSurvey = {
  id: null,
  name: '',
  codeName: '',
  area: '',
  date: '—',
  rawDate: '',
  areaCoveredKm2: 0,
  avgConfidence: 0,
  cleanupProgress: { completed: 0, total: 0 },
  sonarType: 'Side-Scan Sonar',
  status: 'Active',
  startLocation: null,
  depthRange: '0 m',
  vesselName: 'Survey Vessel',
  headingDeg: 0,
  timestamp: '',
};

export const surveysList = [];

export const allDetections = [];

export const recentDetections = [];

export const objectDistribution = [];

export const riskDistribution = {
  high: { count: 0, percent: 0 },
  medium: { count: 0, percent: 0 },
  low: { count: 0, percent: 0 },
};

export const priorityQueue = [];

export const priorityExplanations = {};

export const cleanupRouteStats = {
  totalTargets: 0,
  highPriority: 0,
  estDistance: '0 km',
  estDuration: '0 hrs',
  waypoints: [],
};

export const pipelineStages = [
  'Raw sonar',
  'Preprocess',
  'Denoise',
  'Contrast',
  'Normalize',
  'YOLO AI',
  'Detection'
];

export const stageTelemetryMessages = [
  'Stage 1/7: Reading raw acoustic transducer array...',
  'Stage 2/7: Preprocessing & geometric rectification...',
  'Stage 3/7: Applying bilateral speckle denoising...',
  'Stage 4/7: Enhancing contrast via CLAHE histogram equalization...',
  'Stage 5/7: Normalizing intensity range to [0, 1]...',
  'Stage 6/7: Neural inference · YOLOv8 detection engine...',
  'Stage 7/7: Generating bounding boxes and confidence telemetry...'
];

export const defaultSystemSettings = {
  modelOnline: true,
  modelVersion: 'Aqua Vision 100-Epoch Sonar YOLO · Active',
  modelCheckpoint: 'aqua_vision_100ep_best.pt',
  architecture: 'YOLOv11n + Acoustic Attention Head',
  trainingCurriculum: '100 Epochs (Physics-Constrained)',
  inputResolution: '640 × 640 px (Acoustic rectified)',
  inferenceDevice: 'GPU (CUDA) / CPU fallback',
  confidenceThreshold: 0.15,
  iouThreshold: 0.45,
  preprocessing: {
    swathCorrection: true,
    denoising: true,
    clahe: true,
    normalization: true,
  },
  classes: [
    { id: 0, name: 'Shipwreck', threshold: 0.25, icon: '🚢', rationale: 'Suppresses seafloor reverberation; high precision on structural hulls' },
    { id: 1, name: 'Aircraft', threshold: 0.21, icon: '✈️', rationale: 'Captures aerodynamic geometries and distinct acoustic shadow signatures' },
    { id: 2, name: 'Mine / UXO', threshold: 0.10, icon: '💣', rationale: 'High-sensitivity recall for micro-targets (<20px) with sharp acoustic shadows' },
    { id: 3, name: 'Fishing Gear', threshold: 0.16, icon: '🕸️', rationale: 'Resolves diffuse, low-contrast acoustic reflections from ghost nets & traps' },
  ],
  weights: [
    { label: 'Object type hazard', percent: 40, detail: 'Mine: 1.00 · Shipwreck: 0.90 · Aircraft: 0.85 · Fishing gear: 0.80' },
    { label: 'Detection confidence', percent: 30, detail: 'YOLO posterior class confidence probability' },
    { label: 'Bathymetric depth', percent: 15, detail: 'Normalized against 100 m acoustic reference baseline' },
    { label: 'Acoustic data quality', percent: 15, detail: 'Signal-to-noise ratio & swath clarity factor' },
  ],
};
