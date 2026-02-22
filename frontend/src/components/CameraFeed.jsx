import { useEffect, useRef, useState, useCallback } from 'react';
import { FiCamera, FiCameraOff } from 'react-icons/fi';
import { API_BASE, getPainColor } from '../utils/painLevels';

// MediaPipe Face Mesh connectivity for drawing mesh lines
// Key landmark connections for pain-relevant AUs
const LANDMARK_GROUPS = {
  leftBrow: { indices: [70, 63, 105, 66, 107], color: '#f59e0b' },
  rightBrow: { indices: [336, 296, 334, 293, 300], color: '#f59e0b' },
  leftEye: { indices: [33, 160, 158, 133, 153, 144, 33], color: '#06b6d4' },
  rightEye: { indices: [362, 385, 387, 263, 373, 380, 362], color: '#06b6d4' },
  nose: { indices: [6, 1, 0], color: '#10b981' },
  mouth: { indices: [78, 13, 308, 14, 78], color: '#ec4899' },
};

export default function CameraFeed({ onAnalysis, isActive = true }) {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const processingRef = useRef(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState(null);
  const [faceDetected, setFaceDetected] = useState(false);

  // Send frame to backend for analysis
  const analyzeFrame = useCallback(async (base64Frame) => {
    if (processingRef.current) return;
    processingRef.current = true;

    try {
      const res = await fetch(`${API_BASE}/api/analyze/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: base64Frame }),
      });

      if (!res.ok) {
        processingRef.current = false;
        return;
      }

      const data = await res.json();
      setFaceDetected(data.face_detected);

      // Draw landmarks overlay
      drawLandmarks(data.landmarks, data.facial_score);

      // Pass analysis result up to parent
      if (onAnalysis) {
        onAnalysis(data);
      }
    } catch (err) {
      // Silently handle - backend might not be running
    } finally {
      processingRef.current = false;
    }
  }, [onAnalysis]);

  // Draw landmark overlay on canvas
  const drawLandmarks = useCallback((landmarks, score) => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks || landmarks.length === 0) return;

    // Scale landmarks from backend coordinates to canvas
    const scaleX = canvas.width / (video.videoWidth || 640);
    const scaleY = canvas.height / (video.videoHeight || 480);

    // Draw mesh dots (all landmarks, subtle)
    ctx.fillStyle = 'rgba(6, 182, 212, 0.15)';
    for (const [x, y] of landmarks) {
      ctx.beginPath();
      ctx.arc(x * scaleX, y * scaleY, 1, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw key AU landmark groups (highlighted)
    for (const [groupName, group] of Object.entries(LANDMARK_GROUPS)) {
      ctx.strokeStyle = group.color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let i = 0; i < group.indices.length; i++) {
        const idx = group.indices[i];
        if (idx >= landmarks.length) continue;
        const [x, y] = landmarks[idx];
        const sx = x * scaleX;
        const sy = y * scaleY;

        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      // Draw dots at key points
      ctx.fillStyle = group.color;
      for (const idx of group.indices) {
        if (idx >= landmarks.length) continue;
        const [x, y] = landmarks[idx];
        ctx.beginPath();
        ctx.arc(x * scaleX, y * scaleY, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Pain score badge overlay
    if (score != null) {
      const painColor = getPainColor(score);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.roundRect(10, 10, 130, 40, 8);
      ctx.fill();

      ctx.fillStyle = painColor;
      ctx.font = 'bold 18px Inter, system-ui';
      ctx.fillText(`Pain: ${score.toFixed(1)}`, 20, 37);
    }
  }, []);

  useEffect(() => {
    if (!isActive) return;

    let frameInterval;

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setCameraOn(true);
        }

        // Capture frames at ~3 FPS for ML processing (balance speed vs. load)
        frameInterval = setInterval(() => {
          captureAndAnalyze();
        }, 333);
      } catch (err) {
        setError('Camera access denied. Please allow camera permissions.');
        console.error('Camera error:', err);
      }
    }

    function captureAndAnalyze() {
      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      const ctx = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      const base64 = dataUrl.split(',')[1];
      analyzeFrame(base64);
    }

    startCamera();

    return () => {
      clearInterval(frameInterval);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, [isActive, analyzeFrame]);

  return (
    <div className="relative rounded-lg overflow-hidden bg-slate-900 border border-slate-700">
      <div className="aspect-video relative">
        {/* Video feed */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />

        {/* Landmark overlay canvas */}
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Hidden capture canvas */}
        <canvas ref={captureCanvasRef} className="hidden" />

        {/* Status badges */}
        <div className="absolute top-2 right-2 flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
              faceDetected
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'bg-slate-500/20 text-slate-400'
            }`}
          >
            {faceDetected ? 'Face Detected' : 'No Face'}
          </span>
          <span
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
              cameraOn
                ? 'bg-green-500/20 text-green-400'
                : 'bg-red-500/20 text-red-400'
            }`}
          >
            {cameraOn ? <FiCamera size={12} /> : <FiCameraOff size={12} />}
            {cameraOn ? 'LIVE' : 'OFF'}
          </span>
        </div>

        {/* AU Legend */}
        {faceDetected && (
          <div className="absolute bottom-2 left-2 flex gap-2">
            {[
              { label: 'Brow', color: '#f59e0b' },
              { label: 'Eyes', color: '#06b6d4' },
              { label: 'Nose', color: '#10b981' },
              { label: 'Mouth', color: '#ec4899' },
            ].map((item) => (
              <span
                key={item.label}
                className="flex items-center gap-1 text-[10px] text-slate-300 bg-black/50 px-1.5 py-0.5 rounded"
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                {item.label}
              </span>
            ))}
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80">
            <p className="text-red-400 text-sm text-center px-4">{error}</p>
          </div>
        )}

        {!cameraOn && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80">
            <p className="text-slate-400 text-sm">Initializing camera...</p>
          </div>
        )}
      </div>
    </div>
  );
}
