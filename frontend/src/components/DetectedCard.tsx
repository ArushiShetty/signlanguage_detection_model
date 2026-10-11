import { motion, useReducedMotion } from "framer-motion";

type TopCandidate = {
  label: string;
  confidence: number;
};

type Props = {
  text: string;
  confidence?: number;
  top3?: TopCandidate[];
  holdProgress?: number;
  dimmed?: boolean;
};

export function DetectedCard({
  text,
  confidence = 0,
  top3 = [],
  holdProgress = 0,
  dimmed = false,
}: Props) {
  const reduce = useReducedMotion();
  const confPct = Math.round(confidence * 100);

  return (
    <div className="panel relative overflow-hidden p-6 flex flex-col justify-between min-h-[240px]">
      {/* Ambient Halo Glow on High Confidence */}
      {confidence > 0.5 ? (
        <div
          className="absolute inset-0 pointer-events-none transition-opacity duration-500"
          style={{
            background: `radial-gradient(circle at 50% 45%, rgba(56, 189, 248, ${Math.min(0.2, confidence * 0.22)}) 0%, transparent 70%)`,
          }}
        />
      ) : null}

      <div className="relative z-10 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            {confidence > 0.5 ? (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75" />
            ) : null}
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                confidence > 0.5 ? "bg-sky-400" : "bg-slate-600"
              }`}
            />
          </span>
          <span className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
            Live Detection
          </span>
        </div>

        {confidence > 0 ? (
          <span
            className={`text-xs px-2.5 py-0.5 rounded-full font-medium transition-all ${
              confidence >= 0.85
                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 shadow-sm shadow-emerald-500/10"
                : confidence >= 0.55
                ? "bg-sky-500/15 text-sky-400 border border-sky-500/25 shadow-sm shadow-sky-500/10"
                : "bg-slate-500/15 text-slate-400 border border-slate-500/20"
            }`}
          >
            {confPct}% confidence
          </span>
        ) : null}
      </div>

      {/* Main Big Letter Display */}
      <div className="relative z-10 my-3 flex flex-col items-center justify-center">
        <motion.div
          key={text}
          initial={reduce ? false : { scale: 0.85, opacity: 0.3 }}
          animate={{ scale: 1, opacity: dimmed ? 0.35 : 1 }}
          transition={
            reduce ? { duration: 0 } : { type: "spring", stiffness: 450, damping: 26 }
          }
          className="text-center"
        >
          <span className="font-display text-7xl md:text-8xl font-bold accent-text tracking-tight break-all drop-shadow-sm">
            {text || "—"}
          </span>
        </motion.div>

        {/* Hold progress bar */}
        {holdProgress > 0 && holdProgress < 1 ? (
          <div className="w-40 mt-3">
            <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden p-0.5">
              <div
                className="h-full bg-gradient-to-r from-sky-400 to-indigo-400 rounded-full transition-all duration-75 shadow-sm shadow-sky-400/50"
                style={{ width: `${Math.round(holdProgress * 100)}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-400 font-medium block text-center mt-1">
              Confirming gesture...
            </span>
          </div>
        ) : null}
      </div>

      {/* Top 3 Alternative Candidates Breakdown */}
      {top3 && top3.length > 0 ? (
        <div className="relative z-10 space-y-1.5 pt-3 border-t border-white/5">
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            {top3.slice(0, 3).map((item, i) => (
              <div
                key={item.label + i}
                className="chic-pill rounded-xl px-2.5 py-1.5 flex flex-col gap-1 items-start"
              >
                <div className="w-full flex items-center justify-between">
                  <span className="font-medium text-slate-200">
                    {item.label === "space" ? "Space" : item.label === "del" ? "Del" : item.label}
                  </span>
                  <span className="text-[10px] text-slate-400 font-semibold">
                    {Math.round(item.confidence * 100)}%
                  </span>
                </div>
                {/* Micro Progress Bar */}
                <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-sky-400 to-indigo-400 rounded-full"
                    style={{ width: `${Math.round(item.confidence * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
