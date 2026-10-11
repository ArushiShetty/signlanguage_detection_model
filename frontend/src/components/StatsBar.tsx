type Props = {
  fps: number;
  latency: number;
  connected: boolean;
};

export function StatsBar({ fps, latency, connected }: Props) {
  return (
    <div className="chic-pill flex items-center gap-2.5 px-3 py-1.5 rounded-xl text-xs text-slate-300 select-none">
      <span className="flex items-center gap-1.5 font-medium">
        <span
          className={`h-2 w-2 rounded-full ${
            connected ? "bg-emerald-400" : "bg-rose-400 animate-pulse"
          }`}
          aria-label={connected ? "Connected" : "Disconnected"}
        />
        <span className="text-[11px] text-slate-400 font-mono">
          {connected ? "LIVE" : "OFFLINE"}
        </span>
      </span>
      <span className="text-white/20">|</span>
      <span className="font-mono text-[11px]">
        {fps.toFixed(0)} <span className="text-slate-400">FPS</span>
      </span>
      <span className="text-white/20">|</span>
      <span className="font-mono text-[11px]">
        {latency.toFixed(0)} <span className="text-slate-400">ms</span>
      </span>
    </div>
  );
}
