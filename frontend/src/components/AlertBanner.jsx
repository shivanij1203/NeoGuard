import { useEffect, useRef } from 'react';
import { FiAlertTriangle, FiX } from 'react-icons/fi';
import { getPainLevel, isUrgent } from '../utils/painLevels';

export default function AlertBanner({ alerts, onDismiss }) {
  const audioRef = useRef(null);

  useEffect(() => {
    // Play alert sound for urgent alerts
    const hasUrgent = alerts.some((a) => isUrgent(a.score));
    if (hasUrgent && audioRef.current) {
      audioRef.current.play().catch(() => {});
    }
  }, [alerts]);

  if (!alerts || alerts.length === 0) return null;

  return (
    <div className="space-y-2">
      {/* Hidden audio element for alert sound */}
      <audio ref={audioRef} preload="auto">
        <source src="data:audio/wav;base64,UklGRl9vT19teleaf/8=" type="audio/wav" />
      </audio>

      {alerts.map((alert, idx) => {
        const level = getPainLevel(alert.score);
        const urgent = isUrgent(alert.score);

        return (
          <div
            key={alert.id || idx}
            className={`flex items-center justify-between px-4 py-3 rounded-lg border ${
              urgent
                ? 'bg-red-500/10 border-red-500/30 alert-pulse'
                : 'bg-orange-500/10 border-orange-500/30'
            }`}
          >
            <div className="flex items-center gap-3">
              <FiAlertTriangle
                className={urgent ? 'text-red-400' : 'text-orange-400'}
                size={20}
              />
              <div>
                <p className={`font-semibold text-sm ${urgent ? 'text-red-400' : 'text-orange-400'}`}>
                  {urgent ? 'URGENT' : 'ALERT'}: {level.label} — Bed {alert.bedNumber}
                </p>
                <p className="text-xs text-slate-400">
                  Patient: {alert.patientName} | Score: {alert.score.toFixed(1)} | {alert.timestamp}
                </p>
              </div>
            </div>
            {onDismiss && (
              <button
                onClick={() => onDismiss(alert.id || idx)}
                className="text-slate-400 hover:text-white p-1"
              >
                <FiX size={16} />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
