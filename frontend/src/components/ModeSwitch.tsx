import { motion, useReducedMotion } from "framer-motion";
import type { Mode } from "../lib/ws";

type Props = {
  mode: Mode;
  onChange: (mode: Mode) => void;
};

export function ModeSwitch({ mode, onChange }: Props) {
  return (
    <Seg
      ariaLabel="Recognition mode"
      value={mode}
      onChange={onChange}
      options={[
        { id: "alphabet", label: "Alphabet" },
        { id: "phrase", label: "Phrases" },
      ]}
    />
  );
}

export function Seg<T extends string>({
  value,
  onChange,
  options,
  ariaLabel,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { id: T; label: string }[];
  ariaLabel: string;
}) {
  const reduce = useReducedMotion();
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className="inline-flex rounded-xl border border-white/10 bg-slate-900/60 backdrop-blur-md p-1 shadow-inner"
    >
      {options.map((o) => {
        const active = value === o.id;
        return (
          <button
            key={o.id}
            role="tab"
            aria-selected={active}
            className={`relative rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-colors ${
              active ? "text-white" : "text-slate-400 hover:text-slate-200"
            }`}
            onClick={() => onChange(o.id)}
          >
            {active && (
              <motion.div
                layoutId={`active-${ariaLabel}`}
                transition={{ type: "spring", stiffness: 500, damping: 35 }}
                className="absolute inset-0 rounded-lg bg-gradient-to-r from-sky-500/25 to-indigo-500/25 border border-sky-400/40 shadow-sm shadow-sky-500/20"
              />
            )}
            <span className="relative z-10">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}
