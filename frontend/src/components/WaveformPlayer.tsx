"use client";
import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

interface Props {
  url: string;
  label: string;
  color?: string;
}

export default function WaveformPlayer({ url, label, color = "#a3e635" }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    wsRef.current?.destroy();
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: color + "66",
      progressColor: color,
      height: 64,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      url,
    });
    ws.on("ready", () => { setReady(true); setDuration(ws.getDuration()); });
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => setPlaying(false));
    ws.on("timeupdate", (t) => setCurrentTime(t));
    wsRef.current = ws;
    return () => ws.destroy();
  }, [url, color]);

  const fmt = (s: number) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  return (
    <div className="bg-white/5 rounded-xl p-4 border border-white/10">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-white/60">{label}</span>
        <span className="text-xs text-white/40">{fmt(currentTime)} / {fmt(duration)}</span>
      </div>
      <div ref={containerRef} className="w-full" />
      <div className="flex gap-3 mt-3">
        <button
          onClick={() => wsRef.current?.playPause()}
          disabled={!ready}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-lime-500/20 hover:bg-lime-500/30 disabled:opacity-40 text-lime-400 text-sm font-medium transition-colors"
        >
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <button
          onClick={() => { wsRef.current?.seekTo(0); wsRef.current?.pause(); }}
          disabled={!ready}
          className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 disabled:opacity-40 text-white/60 text-sm transition-colors"
        >
          ⏮
        </button>
      </div>
    </div>
  );
}
