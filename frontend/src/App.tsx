import { useEffect, useState } from "react";
import { Moon, Sun, Activity } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { Onboarding } from "./components/Onboarding";
import { WebcamStage } from "./components/WebcamStage";
import { DetectedCard } from "./components/DetectedCard";
import { ConfidenceRing } from "./components/ConfidenceRing";
import { TranslationPanel } from "./components/TranslationPanel";
import { ModeSwitch } from "./components/ModeSwitch";
import { ModelSwitch } from "./components/ModelSwitch";
import { StatsBar } from "./components/StatsBar";
import { ReferenceChart } from "./components/ReferenceChart";
import { CalibratorModal } from "./components/CalibratorModal";
import { useWebcam } from "./hooks/useWebcam";
import { useSignSocket } from "./hooks/useSignSocket";
import type { Mode, ModelName } from "./lib/ws";

export function App() {
  const reduce = useReducedMotion();
  const { videoRef, ready, error, start } = useWebcam();
  const [mode, setMode] = useState<Mode>("alphabet");
  const [model, setModel] = useState<ModelName>("cnn");
  const [dark, setDark] = useState(true);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [calibratorOpen, setCalibratorOpen] = useState(false);
  const sock = useSignSocket({ videoRef, enabled: ready, mode, model });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  if (!ready) {
    return (
      <>
        <Orbs />
        <Onboarding error={error} onEnable={start} />
      </>
    );
  }

  const pred = sock.prediction?.prediction;
  const display = pred?.display ?? "—";
  const conf = pred?.confidence ?? 0;

  return (
    <div className="relative min-h-screen aurora-bg text-slate-100 selection:bg-sky-500/30 p-3 sm:p-5 md:p-8">
      <Orbs />

      {/* Pinterest-Style Floating Showcase Canvas */}
      <div className="relative z-10 mx-auto max-w-7xl showcase-canvas glow-rim p-5 sm:p-7 md:p-8 space-y-6">
        {/* Chic Glass Navbar */}
        <motion.header
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-wrap items-center justify-between gap-4 pb-5 border-b border-white/10"
        >
          <div className="flex items-center gap-3.5">
            <div className="relative flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-pink-400 via-indigo-500 to-sky-400 p-0.5 shadow-lg shadow-indigo-500/25">
              <div className="flex h-full w-full items-center justify-center rounded-[14px] bg-slate-950/90 backdrop-blur-md">
                <span className="font-display text-xl font-bold accent-text">S</span>
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="font-display text-2xl md:text-3xl font-bold text-white tracking-tight">
                  SignSpeak
                </span>
                <span className="inline-block rounded-full bg-gradient-to-r from-pink-500/15 via-indigo-500/15 to-sky-500/15 px-3 py-0.5 text-[10px] font-bold uppercase tracking-widest text-sky-300 border border-sky-400/30 shadow-sm">
                  Neural Studio
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium">Real-Time ASL Alphabet & Dynamic Motion Translator</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <ModeSwitch mode={mode} onChange={setMode} />
            <ModelSwitch model={model} onChange={setModel} disabled={mode !== "alphabet"} />
          </div>

          <div className="flex items-center gap-3">
            <StatsBar fps={sock.fps} latency={sock.latency} connected={sock.connected} />

            <button
              type="button"
              onClick={() => setCalibratorOpen(true)}
              title="Open Calibrator Studio"
              className="chic-pill flex items-center gap-2 rounded-xl px-3.5 py-2 text-xs font-semibold text-slate-300 hover:text-white transition-colors"
            >
              <Activity size={15} className="text-sky-400" />
              <span className="hidden sm:inline">Diagnostics</span>
            </button>

            <motion.button
              type="button"
              aria-label="Toggle light and dark"
              whileTap={reduce ? undefined : { scale: 0.95 }}
              className="chic-pill rounded-xl p-2.5 text-slate-300 hover:text-white transition-colors"
              onClick={() => setDark((d) => !d)}
            >
              {dark ? <Sun size={16} /> : <Moon size={16} />}
            </motion.button>
          </div>
        </motion.header>

        {/* Main Workbench Grid */}
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] items-start">
          {/* Left: Video Stage with HUD */}
          <motion.div
            initial={{ scale: 0.97, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <WebcamStage
              videoRef={videoRef}
              hands={sock.hands}
              pose={sock.pose}
              bbox={sock.bbox}
              label={pred?.label}
              connected={sock.connected}
              handDetected={Boolean(sock.prediction?.hand_detected)}
            />
          </motion.div>

          {/* Right: Detected Card & Sentence Builder */}
          <motion.div
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="space-y-5 flex flex-col justify-between"
          >
            <div className="grid grid-cols-[1fr_auto] gap-3 items-stretch">
              <DetectedCard
                text={display}
                confidence={conf}
                top3={sock.prediction?.top3}
                holdProgress={sock.holdProgress}
                dimmed={conf > 0 && conf < 0.45}
              />
              <div className="panel p-4 flex items-center justify-center">
                <ConfidenceRing confidence={conf} hold={sock.holdProgress} />
              </div>
            </div>

            <TranslationPanel
              sentence={sock.sentence}
              committed={sock.committed?.token ?? null}
              onClear={sock.sendReset}
              onBackspace={sock.sendBackspace}
              autoSpeak={autoSpeak}
              onAutoSpeak={setAutoSpeak}
            />
          </motion.div>
        </div>

        {/* Bottom Section: Dictionary Chart & Interactive Streak Challenge */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <ReferenceChart lastCommitted={sock.committed?.token ?? null} mode={mode} />
        </motion.div>

        {/* Calibrator Modal */}
        <CalibratorModal
          isOpen={calibratorOpen}
          onClose={() => setCalibratorOpen(false)}
          currentResult={sock.prediction}
          mode={mode}
        />
      </div>
    </div>
  );
}

function Orbs() {
  return (
    <>
      <div className="bg-orb left-[-10%] top-[-10%] bg-sky-500/25" />
      <div
        className="bg-orb right-[-10%] bottom-[-15%] bg-indigo-600/25"
        style={{ animationDelay: "6s" }}
      />
      <div
        className="bg-orb left-[35%] top-[40%] bg-purple-600/20"
        style={{ animationDelay: "12s" }}
      />
    </>
  );
}
