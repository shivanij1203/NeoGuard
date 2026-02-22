import { useState, useEffect, useCallback } from 'react';
import PatientCard from './PatientCard';
import PainGauge from './PainGauge';
import PainChart from './PainChart';
import AlertBanner from './AlertBanner';
import CameraFeed from './CameraFeed';
import { shouldAlert, API_BASE } from '../utils/painLevels';
import { FiPlus, FiActivity } from 'react-icons/fi';

export default function Dashboard() {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [painHistory, setPainHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [latestAnalysis, setLatestAnalysis] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);

  // Check backend health
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        setBackendOnline(res.ok);
      } catch {
        setBackendOnline(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Fetch patients
  useEffect(() => {
    fetchPatients();
  }, []);

  async function fetchPatients() {
    try {
      const res = await fetch(`${API_BASE}/api/patients/`);
      const data = await res.json();
      setPatients(data);
      if (data.length > 0 && !selectedPatient) {
        setSelectedPatient(data[0]);
      }
    } catch (err) {
      console.error('Failed to fetch patients:', err);
    }
  }

  // Handle analysis results from CameraFeed
  const handleAnalysis = useCallback(
    (data) => {
      setLatestAnalysis(data);

      // Add to pain history chart
      setPainHistory((prev) => {
        const newPoint = {
          time: new Date(data.timestamp).toLocaleTimeString(),
          score: data.composite_score,
          facial: data.facial_score,
          audio: data.audio_score,
        };
        return [...prev, newPoint].slice(-120); // Keep last 120 readings (~2 min at 3fps)
      });

      // Generate alert if threshold crossed
      if (shouldAlert(data.composite_score) && selectedPatient) {
        setAlerts((prev) => {
          // Debounce: don't add if same patient alerted in last 5 seconds
          const recentSame = prev.find(
            (a) => a.patientName === selectedPatient.name && Date.now() - a.id < 5000
          );
          if (recentSame) return prev;

          return [
            {
              id: Date.now(),
              patientName: selectedPatient.name,
              bedNumber: selectedPatient.bed_number,
              score: data.composite_score,
              timestamp: new Date(data.timestamp).toLocaleTimeString(),
            },
            ...prev.slice(0, 9),
          ];
        });
      }
    },
    [selectedPatient]
  );

  // Add patient
  async function handleAddPatient(e) {
    e.preventDefault();
    const form = new FormData(e.target);
    try {
      await fetch(`${API_BASE}/api/patients/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: form.get('name'),
          bed_number: form.get('bed_number'),
          gestational_age_weeks: form.get('gestational_age')
            ? parseInt(form.get('gestational_age'))
            : null,
          birth_weight_grams: form.get('birth_weight')
            ? parseInt(form.get('birth_weight'))
            : null,
        }),
      });
      setShowAddModal(false);
      fetchPatients();
    } catch (err) {
      console.error('Failed to add patient:', err);
    }
  }

  function dismissAlert(alertId) {
    setAlerts((prev) => prev.filter((a) => a.id !== alertId));
  }

  const latestScore =
    painHistory.length > 0 ? painHistory[painHistory.length - 1] : null;

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3 bg-slate-900 border-b border-slate-700">
        <div>
          <h1 className="text-xl font-bold text-white">NICU Pain Monitor</h1>
          <p className="text-xs text-slate-400">
            Real-time neonatal pain detection
          </p>
        </div>
        <div className="flex items-center gap-4">
          <span
            className={`flex items-center gap-1 text-xs ${
              backendOnline ? 'text-green-400' : 'text-red-400'
            }`}
          >
            <FiActivity size={14} />
            {backendOnline ? 'Backend Online' : 'Backend Offline'}
          </span>
          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-1 px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm transition-colors"
          >
            <FiPlus size={14} />
            Add Patient
          </button>
        </div>
      </header>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="px-6 py-2">
          <AlertBanner alerts={alerts} onDismiss={dismissAlert} />
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Patient cards sidebar */}
          <div className="col-span-3 space-y-3">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Patients ({patients.length})
            </h2>
            {patients.map((patient) => (
              <PatientCard
                key={patient.id}
                patient={patient}
                latestScore={
                  selectedPatient?.id === patient.id && latestScore
                    ? {
                        composite_score: latestScore.score,
                        facial_score: latestScore.facial,
                        audio_score: latestScore.audio,
                        timestamp: new Date().toISOString(),
                      }
                    : null
                }
                onClick={() => {
                  setSelectedPatient(patient);
                  setPainHistory([]);
                  setLatestAnalysis(null);
                }}
              />
            ))}
            {patients.length === 0 && (
              <div className="text-center py-8 text-slate-500 text-sm">
                No patients yet. Click "Add Patient" to begin monitoring.
              </div>
            )}
          </div>

          {/* Main monitoring area */}
          <div className="col-span-9 space-y-6">
            {selectedPatient ? (
              <>
                {/* Patient header */}
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-bold text-white">
                      {selectedPatient.name}
                    </h2>
                    <p className="text-sm text-slate-400">
                      Bed {selectedPatient.bed_number}
                      {selectedPatient.gestational_age_weeks &&
                        ` | ${selectedPatient.gestational_age_weeks} weeks GA`}
                    </p>
                  </div>
                  <PainGauge score={latestScore?.score ?? 0} size={100} />
                </div>

                {/* Camera + Score details */}
                <div className="grid grid-cols-2 gap-6">
                  <CameraFeed
                    onAnalysis={handleAnalysis}
                    isActive={backendOnline}
                  />

                  <div className="space-y-4">
                    {/* Individual scores */}
                    <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
                      <h3 className="text-sm font-semibold text-slate-400 mb-3">
                        Score Breakdown
                      </h3>
                      <div className="space-y-3">
                        <ScoreBar
                          label="Facial"
                          score={latestScore?.facial}
                          max={10}
                          color="#06b6d4"
                        />
                        <ScoreBar
                          label="Audio"
                          score={latestScore?.audio}
                          max={10}
                          color="#8b5cf6"
                        />
                        <ScoreBar
                          label="Composite"
                          score={latestScore?.score}
                          max={10}
                          color="#f97316"
                        />
                      </div>
                    </div>

                    {/* AU Feature Details */}
                    {latestAnalysis?.features && (
                      <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
                        <h3 className="text-sm font-semibold text-slate-400 mb-3">
                          Action Unit Features
                        </h3>
                        <div className="space-y-2 text-xs">
                          <FeatureRow
                            label="AU4 Brow Furrow"
                            value={latestAnalysis.features.brow_eye_dist_norm}
                            low={0.04}
                            high={0.08}
                            inverted
                          />
                          <FeatureRow
                            label="AU6+7 Eye Squeeze"
                            value={latestAnalysis.features.avg_ear}
                            low={0.15}
                            high={0.35}
                            inverted
                          />
                          <FeatureRow
                            label="AU9+10 Nasolabial"
                            value={latestAnalysis.features.nose_lip_dist_norm}
                            low={0.04}
                            high={0.08}
                            inverted
                          />
                          <FeatureRow
                            label="AU27 Mouth Stretch"
                            value={latestAnalysis.features.mouth_aspect_ratio}
                            low={0.1}
                            high={0.6}
                          />
                        </div>
                      </div>
                    )}

                    {/* Detection status */}
                    <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
                      <h3 className="text-sm font-semibold text-slate-400 mb-3">
                        Detection Status
                      </h3>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <StatusIndicator
                          label="Face"
                          active={latestAnalysis?.face_detected}
                        />
                        <StatusIndicator
                          label="Cry"
                          active={latestAnalysis?.cry_detected}
                        />
                        <StatusIndicator label="Camera" active={true} />
                        <StatusIndicator
                          label="Backend"
                          active={backendOnline}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Pain trend chart */}
                <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
                  <h3 className="text-sm font-semibold text-slate-400 mb-3">
                    Pain Trend ({painHistory.length} readings)
                  </h3>
                  <PainChart data={painHistory} height={220} />
                </div>
              </>
            ) : (
              <div className="flex items-center justify-center h-96 text-slate-500">
                <div className="text-center">
                  <FiActivity size={48} className="mx-auto mb-4 opacity-30" />
                  <p>Select a patient or add a new one to begin monitoring</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Add Patient Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 w-96">
            <h2 className="text-lg font-bold text-white mb-4">Add Patient</h2>
            <form onSubmit={handleAddPatient} className="space-y-3">
              <input
                name="name"
                placeholder="Patient Name"
                required
                className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm focus:border-cyan-500 focus:outline-none"
              />
              <input
                name="bed_number"
                placeholder="Bed Number"
                required
                className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm focus:border-cyan-500 focus:outline-none"
              />
              <input
                name="gestational_age"
                type="number"
                placeholder="Gestational Age (weeks)"
                className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm focus:border-cyan-500 focus:outline-none"
              />
              <input
                name="birth_weight"
                type="number"
                placeholder="Birth Weight (grams)"
                className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm focus:border-cyan-500 focus:outline-none"
              />
              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="flex-1 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 px-3 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm"
                >
                  Add
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

function ScoreBar({ label, score, max, color }) {
  const pct = score != null ? (score / max) * 100 : 0;
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label}</span>
        <span style={{ color }}>
          {score != null ? score.toFixed(1) : '\u2014'}
        </span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function FeatureRow({ label, value, low, high, inverted = false }) {
  if (value == null) return null;

  // Calculate pain intensity (0 = no pain, 1 = max pain)
  let ratio;
  if (inverted) {
    ratio = Math.max(0, Math.min(1, (high - value) / (high - low)));
  } else {
    ratio = Math.max(0, Math.min(1, (value - low) / (high - low)));
  }

  const r = Math.round(ratio * 239 + 16);
  const g = Math.round((1 - ratio) * 200 + 40);
  const barColor = `rgb(${r}, ${g}, 40)`;

  return (
    <div>
      <div className="flex justify-between mb-0.5">
        <span className="text-slate-400">{label}</span>
        <span className="text-slate-300 font-mono">{value.toFixed(3)}</span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${ratio * 100}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  );
}

function StatusIndicator({ label, active }) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-2 h-2 rounded-full ${
          active ? 'bg-green-400' : 'bg-slate-600'
        }`}
      />
      <span className="text-slate-300">{label}</span>
    </div>
  );
}
