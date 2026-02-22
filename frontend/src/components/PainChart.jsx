import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart,
} from 'recharts';
import { getPainColor } from '../utils/painLevels';

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const data = payload[0].payload;
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 text-sm">
      <p className="text-slate-300">{data.time}</p>
      <p style={{ color: getPainColor(data.score) }} className="font-bold">
        Score: {data.score.toFixed(1)}
      </p>
      {data.facial != null && (
        <p className="text-slate-400">Facial: {data.facial.toFixed(1)}</p>
      )}
      {data.audio != null && (
        <p className="text-slate-400">Audio: {data.audio.toFixed(1)}</p>
      )}
    </div>
  );
}

export default function PainChart({ data, height = 200 }) {
  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-slate-500 text-sm"
        style={{ height }}
      >
        Waiting for pain score data...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
        <defs>
          <linearGradient id="painGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="time"
          stroke="#64748b"
          tick={{ fontSize: 11 }}
        />
        <YAxis
          domain={[0, 10]}
          stroke="#64748b"
          tick={{ fontSize: 11 }}
          ticks={[0, 2, 4, 6, 8, 10]}
        />
        <Tooltip content={<CustomTooltip />} />
        {/* Alert threshold line */}
        <ReferenceLine y={4} stroke="#f97316" strokeDasharray="5 5" label="" />
        <ReferenceLine y={7} stroke="#ef4444" strokeDasharray="5 5" label="" />
        {/* Pain score area */}
        <Area
          type="monotone"
          dataKey="score"
          stroke="#06b6d4"
          fill="url(#painGradient)"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
