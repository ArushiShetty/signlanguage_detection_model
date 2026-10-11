import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Copy, Trash2, Volume2, Delete, Check, Sparkles, Sliders } from "lucide-react";
import { speak } from "../lib/speech";

type Props = {
  sentence: string;
  committed: string | null;
  onClear: () => void;
  onBackspace: () => void;
  autoSpeak: boolean;
  onAutoSpeak: (v: boolean) => void;
};

function formatNaturalSentence(text: string): string {
  if (!text.trim()) return "";
  let clean = text.trim();
  // Capitalize first letter
  clean = clean.charAt(0).toUpperCase() + clean.slice(1);
  // Add period if no ending punctuation
  if (!/[.!?]$/.test(clean)) {
    clean += ".";
  }
  return clean;
}

export function TranslationPanel({
  sentence,
  committed,
  onClear,
  onBackspace,
  autoSpeak,
  onAutoSpeak,
}: Props) {
  const reduce = useReducedMotion();
  const [flash, setFlash] = useState(false);
  const [copied, setCopied] = useState(false);
  const [naturalMode, setNaturalMode] = useState(true);
  const [showVoiceSettings, setShowVoiceSettings] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [speechRate, setSpeechRate] = useState(1);

  // Load available speech synthesis voices
  useEffect(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      const updateVoices = () => {
        const v = window.speechSynthesis.getVoices();
        setVoices(v);
        if (v.length > 0 && !selectedVoice) {
          const eng = v.find((voice) => voice.lang.startsWith("en")) || v[0];
          setSelectedVoice(eng.name);
        }
      };
      updateVoices();
      window.speechSynthesis.onvoiceschanged = updateVoices;
    }
  }, [selectedVoice]);

  useEffect(() => {
    if (!committed) return;
    setFlash(true);
    const t = window.setTimeout(() => setFlash(false), 500);
    if (autoSpeak) {
      handleSpeak(committed);
    }
    return () => window.clearTimeout(t);
  }, [committed, autoSpeak]);

  const handleSpeak = (textToSpeak: string) => {
    if (!textToSpeak.trim()) return;
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.rate = speechRate;
      if (selectedVoice) {
        const voiceObj = voices.find((v) => v.name === selectedVoice);
        if (voiceObj) utterance.voice = voiceObj;
      }
      window.speechSynthesis.speak(utterance);
    } else {
      speak(textToSpeak);
    }
  };

  const copy = async () => {
    const textToCopy = naturalMode ? formatNaturalSentence(sentence) : sentence;
    if (!textToCopy) return;
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  };

  const displayText = naturalMode && sentence ? formatNaturalSentence(sentence) : sentence;

  return (
    <div className="panel p-5 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
          Sentence Translation
        </span>
        <button
          type="button"
          onClick={() => setNaturalMode((n) => !n)}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs transition-colors ${
            naturalMode
              ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-medium"
              : "chic-pill text-slate-400 hover:text-white"
          }`}
          title="Toggle natural punctuation & casing"
        >
          <Sparkles size={12} />
          <span>{naturalMode ? "Natural Phrasing On" : "Raw Signs"}</span>
        </button>
      </div>

      {/* Main Sentence View */}
      <div className="min-h-[3.5rem] flex items-center bg-black/20 rounded-2xl px-4 py-3 border border-white/5">
        <p className="font-display text-xl md:text-2xl text-slate-100 break-words leading-relaxed">
          {displayText || <span className="text-slate-500 font-normal text-lg">Signs will form your sentence here...</span>}
          {flash ? (
            <motion.span
              className="ml-1 text-sky-400 font-bold"
              initial={reduce ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              ▍
            </motion.span>
          ) : null}
        </p>
      </div>

      {/* Action Buttons Toolbar */}
      <div className="flex flex-wrap items-center gap-2">
        <IconBtn
          label="Speak aloud"
          onClick={() => handleSpeak(displayText || sentence)}
          disabled={!sentence}
        >
          <div className="flex items-center gap-1.5">
            <Volume2 size={15} />
            <span>Speak</span>
            {sentence ? (
              <div className="flex items-center gap-0.5 ml-1">
                <span className="soundwave-bar" style={{ animationDelay: "0ms" }} />
                <span className="soundwave-bar" style={{ animationDelay: "150ms" }} />
                <span className="soundwave-bar" style={{ animationDelay: "300ms" }} />
              </div>
            ) : null}
          </div>
        </IconBtn>

        <IconBtn
          label="Copy to clipboard"
          onClick={copy}
          disabled={!sentence}
        >
          {copied ? <Check size={15} className="text-emerald-400" /> : <Copy size={15} />}
          {copied ? "Copied" : "Copy"}
        </IconBtn>

        <IconBtn
          label="Backspace"
          onClick={onBackspace}
          disabled={!sentence}
        >
          <Delete size={15} /> Backspace
        </IconBtn>

        <IconBtn
          label="Clear all"
          onClick={onClear}
          disabled={!sentence}
        >
          <Trash2 size={15} /> Clear
        </IconBtn>

        <button
          type="button"
          onClick={() => setShowVoiceSettings((v) => !v)}
          className="chic-pill rounded-xl p-2 ml-auto text-slate-400 hover:text-white transition-colors"
          title="Voice and audio settings"
        >
          <Sliders size={15} />
        </button>
      </div>

      {/* Voice & Auto-Speak Settings drawer */}
      {showVoiceSettings ? (
        <div className="p-3.5 rounded-2xl bg-black/25 border border-white/5 space-y-3 text-xs">
          <div className="flex items-center justify-between">
            <span className="font-medium text-slate-300">Voice Persona</span>
            <select
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
              className="bg-slate-900 border border-white/10 text-slate-200 rounded-lg px-2 py-1 max-w-[200px] text-xs focus:outline-none"
            >
              {voices.map((v) => (
                <option key={v.name} value={v.name}>
                  {v.name.slice(0, 24)} ({v.lang})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center justify-between">
            <span className="font-medium text-slate-300">Speech Rate</span>
            <input
              type="range"
              min="0.6"
              max="1.4"
              step="0.1"
              value={speechRate}
              onChange={(e) => setSpeechRate(parseFloat(e.target.value))}
              className="w-32 accent-sky-400"
            />
          </div>
        </div>
      ) : null}

      <div className="flex items-center justify-between text-xs text-slate-400 pt-1">
        <label className="flex items-center gap-2 cursor-pointer hover:text-slate-200 transition-colors select-none">
          <input
            type="checkbox"
            checked={autoSpeak}
            onChange={(e) => onAutoSpeak(e.target.checked)}
            className="rounded accent-sky-400"
          />
          Auto-speak recognized phrases aloud
        </label>
      </div>
    </div>
  );
}

function IconBtn({
  children,
  onClick,
  label,
  disabled = false,
}: {
  children: React.ReactNode;
  onClick: () => void;
  label: string;
  disabled?: boolean;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.button
      type="button"
      aria-label={label}
      disabled={disabled}
      whileTap={reduce || disabled ? undefined : { scale: 0.96 }}
      onClick={onClick}
      className={`chic-pill inline-flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-medium transition-opacity ${
        disabled
          ? "opacity-30 cursor-not-allowed"
          : "hover:bg-white/10 text-slate-200 cursor-pointer"
      }`}
    >
      {children}
    </motion.button>
  );
}
