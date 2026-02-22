import PainGauge from './PainGauge';
import { getPainLevel, shouldAlert } from '../utils/painLevels';
import { FiUser, FiClock } from 'react-icons/fi';

export default function PatientCard({ patient, latestScore, onClick }) {
  const score = latestScore?.composite_score ?? 0;
  const painLevel = getPainLevel(score);
  const alerting = shouldAlert(score);

  return (
    <div
      onClick={onClick}
      className={`bg-slate-800 rounded-xl border p-4 cursor-pointer transition-all hover:bg-slate-750 ${
        alerting
          ? 'border-orange-500/50 shadow-lg shadow-orange-500/10'
          : 'border-slate-700 hover:border-slate-600'
      }`}
    >
      {/* Patient info header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center">
            <FiUser className="text-cyan-400" size={14} />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">{patient.name}</h3>
            <p className="text-xs text-slate-400">Bed {patient.bed_number}</p>
          </div>
        </div>
        <span
          className={`px-2 py-0.5 rounded-full text-xs font-medium ${
            alerting ? 'bg-orange-500/20 text-orange-400' : 'bg-green-500/20 text-green-400'
          }`}
        >
          {alerting ? 'ALERT' : 'STABLE'}
        </span>
      </div>

      {/* Pain gauge */}
      <div className="flex justify-center">
        <PainGauge score={score} size={120} />
      </div>

      {/* Details */}
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-400">
        <div>
          <span className="text-slate-500">Facial:</span>{' '}
          {latestScore?.facial_score?.toFixed(1) ?? '—'}
        </div>
        <div>
          <span className="text-slate-500">Audio:</span>{' '}
          {latestScore?.audio_score?.toFixed(1) ?? '—'}
        </div>
        <div className="flex items-center gap-1">
          <FiClock size={10} />
          {latestScore?.timestamp
            ? new Date(latestScore.timestamp).toLocaleTimeString()
            : '—'}
        </div>
        <div>
          <span className="text-slate-500">Cry:</span>{' '}
          {latestScore?.cry_type ?? 'none'}
        </div>
      </div>
    </div>
  );
}
