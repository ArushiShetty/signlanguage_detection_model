import { motion, useReducedMotion } from "framer-motion";
import { Camera, Sparkles, ShieldCheck } from "lucide-react";

type Props = {
  error: string | null;
  onEnable: () => void;
};

export function Onboarding({ error, onEnable }: Props) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="min-h-screen aurora-bg flex items-center justify-center p-6"
    >
      <div className="showcase-canvas glow-rim max-w-lg w-full p-8 md:p-10 space-y-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-sky-400">
            <Sparkles size={14} />
            <span>SignSpeak Neural Studio</span>
          </div>
          <h1 className="font-display text-3xl md:text-4xl font-bold accent-text tracking-tight">
            Real-Time Sign Language Translation
          </h1>
        </div>

        <p className="text-sm text-slate-300 leading-relaxed">
          Translate American Sign Language alphabet and dynamic motion phrases directly from your webcam into text and spoken voice.
        </p>

        <div className="space-y-3 py-2 text-xs text-slate-300">
          <div className="flex items-start gap-3 chic-pill p-3.5 rounded-2xl">
            <span className="font-mono font-bold text-sky-400">01</span>
            <span>Allow camera access to activate MediaPipe 21-joint landmark tracking.</span>
          </div>
          <div className="flex items-start gap-3 chic-pill p-3.5 rounded-2xl">
            <span className="font-mono font-bold text-indigo-400">02</span>
            <span>Toggle between Alphabet mode (single-frame handshapes) or Phrases (motion sequences).</span>
          </div>
          <div className="flex items-start gap-3 chic-pill p-3.5 rounded-2xl">
            <span className="font-mono font-bold text-pink-400">03</span>
            <span>Hold your hand steady to commit signs to the natural sentence builder with speech synthesis.</span>
          </div>
        </div>

        <motion.button
          whileTap={reduce ? undefined : { scale: 0.98 }}
          onClick={onEnable}
          className="w-full rounded-2xl py-4 font-semibold text-sm text-slate-950 bg-gradient-to-r from-pink-300 via-indigo-300 to-sky-300 flex items-center justify-center gap-2 hover:opacity-95 transition-all cursor-pointer shadow-xl shadow-indigo-500/20"
        >
          <Camera size={16} />
          Enable Camera Access
        </motion.button>

        <div className="flex items-center justify-center gap-1.5 text-[11px] text-slate-400">
          <ShieldCheck size={13} className="text-emerald-400" />
          <span>Local processing: all video frames are processed privately in real-time.</span>
        </div>

        {error ? (
          <div className="p-3.5 rounded-2xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs text-center" role="alert">
            {error}
          </div>
        ) : null}
      </div>
    </motion.div>
  );
}
