import { useEffect, useRef, useState } from "react";
import { Eye, EyeOff, Palette } from "lucide-react";

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export type SkeletonTheme = "ice" | "pearl" | "lavender" | "emerald";

const THEME_COLORS: Record<SkeletonTheme, { joint: string; bone: string; glow: string; label: string }> = {
  ice: { joint: "#38BDF8", bone: "rgba(129, 140, 248, 0.65)", glow: "rgba(56, 189, 248, 0.4)", label: "Ice Cyan" },
  pearl: { joint: "#F8FAFC", bone: "rgba(203, 213, 225, 0.6)", glow: "rgba(248, 250, 252, 0.3)", label: "Pearl White" },
  lavender: { joint: "#C084FC", bone: "rgba(168, 85, 247, 0.6)", glow: "rgba(192, 132, 252, 0.4)", label: "Lavender" },
  emerald: { joint: "#34D399", bone: "rgba(16, 185, 129, 0.6)", glow: "rgba(52, 211, 153, 0.4)", label: "Emerald" },
};

type Props = {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  hands: number[][][];
  pose: number[][];
  bbox: number[] | null;
  label?: string;
  connected: boolean;
  handDetected: boolean;
};

export function WebcamStage({
  videoRef,
  hands,
  pose,
  bbox,
  label,
  connected,
  handDetected,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [theme, setTheme] = useState<SkeletonTheme>("ice");
  const [showSkeleton, setShowSkeleton] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    let raf = 0;

    const draw = () => {
      const w = video.videoWidth || 640;
      const h = video.videoHeight || 480;
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, w, h);

        if (showSkeleton) {
          const colors = THEME_COLORS[theme];

          // 1. Draw Hand Skeletal Connections
          ctx.lineWidth = 2.5;
          ctx.strokeStyle = colors.bone;
          ctx.shadowBlur = 10;
          ctx.shadowColor = colors.glow;
          ctx.fillStyle = colors.joint;

          for (const pts of hands) {
            // Bones
            for (const [a, b] of HAND_CONNECTIONS) {
              if (!pts[a] || !pts[b]) continue;
              ctx.beginPath();
              ctx.moveTo(pts[a][0] * w, pts[a][1] * h);
              ctx.lineTo(pts[b][0] * w, pts[b][1] * h);
              ctx.stroke();
            }

            // Joints
            for (let idx = 0; idx < pts.length; idx++) {
              const p = pts[idx];
              const radius = idx === 4 || idx === 8 || idx === 12 || idx === 16 || idx === 20 ? 5 : 3.5;
              ctx.beginPath();
              ctx.arc(p[0] * w, p[1] * h, radius, 0, Math.PI * 2);
              ctx.fill();
            }
          }

          // 2. Draw Upper Pose Keypoints (for phrases)
          if (pose.length > 0) {
            ctx.shadowBlur = 8;
            ctx.shadowColor = "rgba(129, 140, 248, 0.4)";
            ctx.fillStyle = "rgba(167, 139, 250, 0.85)";
            for (const p of pose) {
              ctx.beginPath();
              ctx.arc(p[0] * w, p[1] * h, 4.5, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }

        // 3. Chic Minimal Corner Brackets Bounding Box
        if (bbox && bbox.length === 4) {
          const [x, y, bw, bh] = bbox;
          const rx = (x / 320) * w;
          const ry = (y / 240) * h;
          const rw = (bw / 320) * w;
          const rh = (bh / 240) * h;
          const cornerLen = Math.min(22, rw * 0.25, rh * 0.25);

          ctx.shadowBlur = 0;
          ctx.strokeStyle = "#38BDF8";
          ctx.lineWidth = 2.5;

          // Top-Left
          ctx.beginPath();
          ctx.moveTo(rx, ry + cornerLen);
          ctx.lineTo(rx, ry);
          ctx.lineTo(rx + cornerLen, ry);
          ctx.stroke();

          // Top-Right
          ctx.beginPath();
          ctx.moveTo(rx + rw - cornerLen, ry);
          ctx.lineTo(rx + rw, ry);
          ctx.lineTo(rx + rw, ry + cornerLen);
          ctx.stroke();

          // Bottom-Right
          ctx.beginPath();
          ctx.moveTo(rx + rw, ry + rh - cornerLen);
          ctx.lineTo(rx + rw, ry + rh);
          ctx.lineTo(rx + rw - cornerLen, ry + rh);
          ctx.stroke();

          // Bottom-Left
          ctx.beginPath();
          ctx.moveTo(rx + cornerLen, ry + rh);
          ctx.lineTo(rx, ry + rh);
          ctx.lineTo(rx, ry + rh - cornerLen);
          ctx.stroke();

          // Chic Label Pill
          if (label) {
            ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
            const tagW = Math.max(68, label.length * 13 + 24);
            roundRect(ctx, rx, Math.max(4, ry - 32), tagW, 28, 8);
            ctx.fill();

            ctx.fillStyle = "#F8FAFC";
            ctx.font = "600 14px 'Space Grotesk', system-ui, sans-serif";
            ctx.fillText(label, rx + 12, Math.max(22, ry - 13));
          }
        }
      }
      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [videoRef, hands, pose, bbox, label, showSkeleton, theme]);

  const pillText = !connected
    ? "Connecting..."
    : handDetected
    ? "Tracking Active"
    : "Show Hand to Camera";

  const nextTheme = () => {
    const list: SkeletonTheme[] = ["ice", "pearl", "lavender", "emerald"];
    const curr = list.indexOf(theme);
    setTheme(list[(curr + 1) % list.length]);
  };

  return (
    <div className="panel relative overflow-hidden aspect-[4/3] group">
      <video
        ref={videoRef as unknown as React.RefObject<HTMLVideoElement>}
        className="absolute inset-0 h-full w-full object-cover"
        style={{ transform: "scaleX(-1)" }}
        playsInline
        muted
        aria-label="Webcam feed"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full"
        style={{ transform: "scaleX(-1)" }}
        aria-hidden
      />

      {/* Cinematic HUD Laser Scanline */}
      {handDetected ? <div className="hud-scanline" /> : null}

      {/* Top Left Chic Status Badge */}
      <div className="absolute left-4 top-4 flex items-center gap-2">
        <span className="chic-pill flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium text-slate-200">
          <span
            className={`h-2 w-2 rounded-full ${
              !connected
                ? "bg-amber-400 animate-pulse"
                : handDetected
                ? "bg-emerald-400 shadow-sm shadow-emerald-400/50"
                : "bg-slate-400"
            }`}
          />
          {pillText}
        </span>
      </div>

      {/* Top Right Chic Controls */}
      <div className="absolute right-4 top-4 flex items-center gap-2 opacity-80 group-hover:opacity-100 transition-opacity">
        <button
          type="button"
          onClick={nextTheme}
          title={`Theme: ${THEME_COLORS[theme].label}`}
          className="chic-pill flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs text-slate-200 hover:bg-white/10 transition-colors"
        >
          <Palette size={13} />
          <span>{THEME_COLORS[theme].label}</span>
        </button>
        <button
          type="button"
          onClick={() => setShowSkeleton((v) => !v)}
          title={showSkeleton ? "Hide Skeleton" : "Show Skeleton"}
          className="chic-pill rounded-full p-2 text-slate-200 hover:bg-white/10 transition-colors"
        >
          {showSkeleton ? <Eye size={14} /> : <EyeOff size={14} />}
        </button>
      </div>

      {/* Bottom Telemetry Badge */}
      <div className="absolute bottom-4 left-4 flex items-center gap-2">
        <span className="text-[10px] text-slate-300/70 tracking-widest uppercase font-mono chic-pill px-2.5 py-1 rounded-lg">
          {handDetected ? "Neural Skeleton Lock [21-Joint]" : "Vision Sensor Standby"}
        </span>
      </div>
    </div>
  );
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
