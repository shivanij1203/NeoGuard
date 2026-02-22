import { getPainLevel, formatScore } from '../utils/painLevels';

export default function PainGauge({ score, size = 140 }) {
  const painLevel = getPainLevel(score);
  const normalizedScore = Math.min(Math.max(score, 0), 10);
  const percentage = normalizedScore / 10;

  // SVG arc calculations
  const radius = (size - 20) / 2;
  const circumference = Math.PI * radius; // Half circle
  const strokeDashoffset = circumference * (1 - percentage);

  const centerX = size / 2;
  const centerY = size / 2 + 10;

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size * 0.7} viewBox={`0 0 ${size} ${size * 0.7}`}>
        {/* Background arc */}
        <path
          d={`M ${size * 0.1} ${centerY} A ${radius} ${radius} 0 0 1 ${size * 0.9} ${centerY}`}
          fill="none"
          stroke="#334155"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Score arc */}
        <path
          d={`M ${size * 0.1} ${centerY} A ${radius} ${radius} 0 0 1 ${size * 0.9} ${centerY}`}
          fill="none"
          stroke={painLevel.color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          style={{ transition: 'stroke-dashoffset 0.5s ease, stroke 0.3s ease' }}
        />
        {/* Score text */}
        <text
          x={centerX}
          y={centerY - 10}
          textAnchor="middle"
          fill={painLevel.color}
          fontSize={size * 0.25}
          fontWeight="bold"
        >
          {formatScore(score)}
        </text>
        <text
          x={centerX}
          y={centerY + 10}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize={size * 0.08}
        >
          / 10
        </text>
      </svg>
      <span
        className="text-sm font-semibold mt-1"
        style={{ color: painLevel.color }}
      >
        {painLevel.label}
      </span>
    </div>
  );
}
