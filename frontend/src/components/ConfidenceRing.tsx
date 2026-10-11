type Props = {
  confidence: number;
  hold: number;
};

export function ConfidenceRing({ confidence, hold }: Props) {
  const color =
    confidence >= 0.85 ? "#34D399" : confidence >= 0.5 ? "#FBBF24" : "#F87171";
  const r = 36;
  const c = 2 * Math.PI * r;
  const confDash = c * Math.min(1, Math.max(0, confidence));
  const holdDash = c * Math.min(1, Math.max(0, hold));
  return (
    <svg width="96" height="96" viewBox="0 0 96 96" aria-label={`Confidence ${Math.round(confidence * 100)} percent`}>
      <g transform="rotate(-90 48 48)">
        <circle cx="48" cy="48" r={r} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="8" />
        <circle
          cx="48"
          cy="48"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={`${confDash} ${c}`}
          strokeLinecap="round"
        />
        <circle
          cx="48"
          cy="48"
          r={r - 12}
          fill="none"
          stroke="rgba(139,92,246,0.9)"
          strokeWidth="3"
          strokeDasharray={`${holdDash} ${c}`}
          strokeLinecap="round"
        />
      </g>
      <text x="48" y="52" textAnchor="middle" fill="currentColor" fontSize="14">
        {Math.round(confidence * 100)}%
      </text>
    </svg>
  );
}
