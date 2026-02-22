export const PAIN_LEVELS = {
  NONE: { min: 0, max: 1, label: 'No Pain', color: '#22c55e', bgColor: 'bg-green-500', textColor: 'text-green-400' },
  MILD: { min: 2, max: 3, label: 'Mild Discomfort', color: '#eab308', bgColor: 'bg-yellow-500', textColor: 'text-yellow-400' },
  MODERATE: { min: 4, max: 6, label: 'Moderate Pain', color: '#f97316', bgColor: 'bg-orange-500', textColor: 'text-orange-400' },
  SEVERE: { min: 7, max: 10, label: 'Severe Pain', color: '#ef4444', bgColor: 'bg-red-500', textColor: 'text-red-400' },
};

export function getPainLevel(score) {
  if (score <= 1) return PAIN_LEVELS.NONE;
  if (score <= 3) return PAIN_LEVELS.MILD;
  if (score <= 6) return PAIN_LEVELS.MODERATE;
  return PAIN_LEVELS.SEVERE;
}

export function getPainColor(score) {
  return getPainLevel(score).color;
}

export function formatScore(score) {
  return typeof score === 'number' ? score.toFixed(1) : '—';
}

export function shouldAlert(score) {
  return score >= 4;
}

export function isUrgent(score) {
  return score >= 7;
}

export const API_BASE = '';
export const WS_BASE = `ws://${window.location.hostname}:8000`;
