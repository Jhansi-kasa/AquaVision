import { Icons } from './components/Icons.jsx';

export const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: Icons.dashboard, crumb: 'Dashboard', title: 'Mission overview' },
  { id: 'surveys', label: 'Surveys', icon: Icons.surveys, crumb: 'Surveys', title: 'Survey management' },
  { id: 'sonar', label: 'Sonar Analysis', icon: Icons.sonar, crumb: 'Sonar Analysis', title: 'Upload · Process · Detect' },
  { id: 'review', label: 'Detection Review', icon: Icons.review, crumb: 'Detection Review', title: 'Human-in-the-loop verification' },
  { id: 'map', label: 'Risk Map', icon: Icons.map, crumb: 'Risk Map', title: 'Detection & survey track' },
  { id: 'cleanup', label: 'Cleanup Priority', icon: Icons.cleanup, crumb: 'Cleanup Priority', title: 'Prioritized removal queue' },
  { id: 'verify', label: 'Verification', icon: Icons.verify, crumb: 'Verification', title: 'Before / after cleanup' },
  { id: 'reports', label: 'Reports', icon: Icons.reports, crumb: 'Reports', title: 'Survey report export' },
  { id: 'settings', label: 'System Status', icon: Icons.settings, crumb: 'System Status', title: 'Model & risk configuration' },
];

export const RISK_STYLE = {
  high:    { text: 'text-rose-600',    bg: 'bg-rose-50',    bar: 'bg-rose-500',    dot: 'bg-rose-500' },
  medium:  { text: 'text-amber-600',   bg: 'bg-amber-50',   bar: 'bg-amber-500',   dot: 'bg-amber-500' },
  low:     { text: 'text-emerald-600', bg: 'bg-emerald-50', bar: 'bg-emerald-500', dot: 'bg-emerald-500' },
  unknown: { text: 'text-slate-500',   bg: 'bg-slate-100',  bar: 'bg-slate-400',   dot: 'bg-slate-400' },
};

export const BOX_COLORS = { high: '#fb7185', medium: '#f59e0b', low: '#34d399', unknown: '#94a3b8' };

export const RECENT = [];

export const OBJ_DIST = [];

export const SURVEYS = [];

export const MAP_DETECTIONS = [];

export const PRIORITY_Q = [];

export const PRIORITY_WHY = {};

export const PIPELINE_STEPS = ['Raw sonar', 'Preprocess', 'Denoise', 'Contrast', 'Normalize', 'YOLO AI', 'Detection'];

export const DETECTED_OBJS = [];

