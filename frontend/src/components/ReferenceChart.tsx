import { useEffect, useMemo, useState } from "react";
import { motion, useReducedMotion, AnimatePresence } from "framer-motion";
import { Check, Flame, ChevronDown, ChevronUp, RefreshCw, Award } from "lucide-react";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const HINTS: Record<string, string> = {
  A: "Tight fist, thumb alongside index knuckle",
  B: "Flat palm, 4 fingers straight up, thumb tucked",
  C: "Curved cup shape with all fingers and thumb",
  D: "Index finger points straight up, others touch thumb",
  E: "Fingers curled tight down, thumb curled beneath",
  F: "Index & thumb touch in circle, other 3 up (OK sign)",
  G: "Index & thumb point horizontally parallel",
  H: "Index & middle point horizontally together",
  I: "Only pinky extended up, thumb across others",
  J: "Pinky draws a J hook in the air",
  K: "Index up, middle forward, thumb between them",
  L: "L shape: Index straight up, thumb out at 90°",
  M: "Thumb tucked under 3 fingers (index, middle, ring)",
  N: "Thumb tucked under 2 fingers (index, middle)",
  O: "All 4 fingertips touch thumb forming an O",
  P: "Index & middle point downward like an inverted K",
  Q: "Index & thumb point downward like an inverted G",
  R: "Index and middle crossed over each other",
  S: "Tight fist, thumb folded across front of knuckles",
  T: "Thumb tucked between index and middle finger",
  U: "Index and middle extended up held tight together",
  V: "Peace sign: Index and middle extended up spread apart",
  W: "Index, middle, and ring fingers up in W shape",
  X: "Index finger hooked/bent like a trigger",
  Y: "Shaka sign: Thumb out and pinky out, others curled",
  Z: "Index finger traces Z in the air",
  space: "Open flat palm facing downwards",
  del: "Thumb pointing down or wipe back",
  hello: "Flat open hand saluting outward from forehead",
  thank_you: "Fingertips touch chin, then move forward & down",
  sorry: "Fist on chest making circular rubbing motion",
  i_love_you: "Thumb, index, and pinky extended (ILY sign)",
  please: "Flat open hand circles on center of chest",
  yes: "Fist nodding up and down",
  no: "Index and middle fingers tap against thumb",
  help: "Dominant fist on flat open palm, lifting up together",
  good: "Fingertips from chin down into other flat palm",
  bye: "Open hand waving side to side",
};

const PHRASES = [
  "hello",
  "thank_you",
  "sorry",
  "i_love_you",
  "please",
  "yes",
  "no",
  "help",
  "good",
  "bye",
];

type Props = {
  lastCommitted: string | null;
  mode: "alphabet" | "phrase";
};

export function ReferenceChart({ lastCommitted, mode }: Props) {
  const reduce = useReducedMotion();
  const [open, setOpen] = useState(false);
  const [practice, setPractice] = useState(false);
  const [streak, setStreak] = useState(0);
  const [bestStreak, setBestStreak] = useState(0);
  const [hit, setHit] = useState(false);

  const pool = useMemo(() => (mode === "alphabet" ? LETTERS : PHRASES), [mode]);
  const [targetIndex, setTargetIndex] = useState(0);

  const target = pool[targetIndex] || pool[0];
  const targetDisplay = useMemo(
    () => (target === "del" ? "Backspace" : target.replace(/_/g, " ").toUpperCase()),
    [target]
  );

  const pickNextTarget = () => {
    let next = Math.floor(Math.random() * pool.length);
    if (next === targetIndex && pool.length > 1) {
      next = (next + 1) % pool.length;
    }
    setTargetIndex(next);
  };

  useEffect(() => {
    if (!practice || !lastCommitted) return;
    const token = lastCommitted.replace(/ /g, "_").toLowerCase();
    const ok =
      mode === "alphabet"
        ? lastCommitted.toUpperCase() === target
        : token === target.toLowerCase() ||
          lastCommitted.toLowerCase() === target.replace(/_/g, " ").toLowerCase();

    if (ok) {
      setHit(true);
      const newStreak = streak + 1;
      setStreak(newStreak);
      if (newStreak > bestStreak) setBestStreak(newStreak);

      const t = window.setTimeout(() => {
        setHit(false);
        pickNextTarget();
      }, 700);
      return () => window.clearTimeout(t);
    }
  }, [lastCommitted, practice, mode, target, streak, bestStreak]);

  return (
    <div className="panel p-5 space-y-4">
      {/* Header and Mode Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-2 text-sm font-semibold text-slate-200 hover:text-white transition-colors cursor-pointer"
          >
            <span>ASL Reference & Practice</span>
            {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          <span className="text-xs text-slate-400">
            {mode === "alphabet" ? "28 Signs" : "11 Motion Phrases"}
          </span>
        </div>

        {/* Practice Mode Toggle */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => {
              setPractice((p) => !p);
              if (!practice) {
                setStreak(0);
                pickNextTarget();
              }
            }}
            className={`chic-pill px-3 py-1.5 rounded-xl text-xs font-medium flex items-center gap-1.5 transition-colors ${
              practice
                ? "bg-amber-500/15 text-amber-300 border-amber-500/30"
                : "text-slate-300 hover:bg-white/10"
            }`}
          >
            <Flame size={14} className={practice ? "text-amber-400" : "text-slate-400"} />
            <span>{practice ? "Practice Mode On" : "Start Practice Challenge"}</span>
          </button>
        </div>
      </div>

      {/* Interactive Practice Challenge Banner */}
      <AnimatePresence>
        {practice && (
          <motion.div
            initial={reduce ? false : { opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-4 rounded-2xl bg-gradient-to-r from-amber-500/10 via-indigo-500/10 to-sky-500/10 border border-amber-500/20 flex flex-wrap items-center justify-between gap-4">
              <div className="space-y-1">
                <span className="text-[11px] uppercase tracking-wider text-amber-300 font-semibold">
                  Current Target Sign
                </span>
                <div className="flex items-baseline gap-3">
                  <h3 className="font-display text-3xl font-bold text-white tracking-wide">
                    {targetDisplay}
                  </h3>
                  <p className="text-xs text-slate-300 italic">{HINTS[target] || ""}</p>
                </div>
              </div>

              <div className="flex items-center gap-4">
                {hit ? (
                  <motion.div
                    initial={reduce ? false : { scale: 0.5, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="flex items-center gap-1.5 text-emerald-400 font-semibold text-sm"
                  >
                    <Check size={18} />
                    <span>Great Job!</span>
                  </motion.div>
                ) : null}

                <div className="flex items-center gap-3">
                  <div className="chic-pill px-3 py-1.5 rounded-xl text-center">
                    <span className="text-[10px] text-slate-400 block uppercase">Streak</span>
                    <span className="font-bold text-amber-400 text-sm">{streak}</span>
                  </div>
                  {bestStreak > 0 ? (
                    <div className="chic-pill px-3 py-1.5 rounded-xl text-center">
                      <span className="text-[10px] text-slate-400 block uppercase">Best</span>
                      <span className="font-bold text-sky-400 text-sm flex items-center gap-1 justify-center">
                        <Award size={12} /> {bestStreak}
                      </span>
                    </div>
                  ) : null}
                  <button
                    type="button"
                    onClick={pickNextTarget}
                    className="chic-pill p-2 rounded-xl text-slate-400 hover:text-white transition-colors"
                    title="Skip to another sign"
                  >
                    <RefreshCw size={14} />
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Collapsible Reference Grid */}
      {open ? (
        <div className="space-y-4 pt-2 border-t border-white/5">
          {mode === "alphabet" ? (
            <div className="grid grid-cols-4 sm:grid-cols-7 md:grid-cols-9 lg:grid-cols-14 gap-2">
              {LETTERS.map((letter) => (
                <div
                  key={letter}
                  className="chic-pill p-2 rounded-xl text-center hover:bg-white/10 transition-colors group relative cursor-help"
                >
                  <p className="font-display text-xl font-bold text-slate-100">{letter}</p>
                  <p className="text-[9px] text-slate-400 truncate mt-0.5">
                    {HINTS[letter]?.slice(0, 14)}...
                  </p>
                  {/* Chic Hover Tooltip */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-30 w-44 p-2 rounded-xl bg-slate-900 border border-white/15 text-[11px] text-slate-200 text-center shadow-xl pointer-events-none">
                    <p className="font-semibold text-sky-300">{letter}</p>
                    <p className="mt-0.5">{HINTS[letter]}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2.5">
              {PHRASES.map((phrase) => (
                <div
                  key={phrase}
                  className="chic-pill p-3 rounded-2xl space-y-1 hover:bg-white/10 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-display font-semibold text-slate-100 capitalize">
                      {phrase.replace(/_/g, " ")}
                    </span>
                    <span className="text-[10px] text-sky-400/80 bg-sky-400/10 px-2 py-0.5 rounded-full font-medium">
                      Motion
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 leading-snug">{HINTS[phrase] || ""}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
