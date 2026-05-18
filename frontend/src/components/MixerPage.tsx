"use client";
import { useState, useCallback, useRef } from "react";
import FileDropzone from "./FileDropzone";
import WaveformPlayer from "./WaveformPlayer";
import ProcessingStats from "./ProcessingStats";
import ReferenceUpload from "./ReferenceUpload";
import { uploadFiles, getJob, exportUrl, type Job } from "@/lib/api";

type Step = "upload" | "processing" | "done" | "error";

const PRESETS = [
  { value: "default", label: "Default" },
  { value: "hip_hop", label: "Hip-Hop" },
  { value: "rnb", label: "R&B" },
  { value: "pop", label: "Pop" },
] as const;

export default function MixerPage() {
  const [beat, setBeat] = useState<File | null>(null);
  const [vocal, setVocal] = useState<File | null>(null);
  const [preset, setPreset] = useState<string>("default");
  const [step, setStep] = useState<Step>("upload");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string>("");
  const [progress, setProgress] = useState<string>("Uploading…");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  const startPolling = useCallback((jobId: string) => {
    const messages = [
      "Loading audio…",
      "Analysing vocal features…",
      "Deciding processing chain…",
      "Applying vocal chain…",
      "Balancing mix…",
      "Mastering…",
    ];
    let i = 0;
    pollRef.current = setInterval(async () => {
      try {
        const j = await getJob(jobId);
        setJob(j);
        if (j.status === "done") {
          stopPolling();
          setStep("done");
        } else if (j.status === "error") {
          stopPolling();
          setError(j.error_message ?? "Unknown error");
          setStep("error");
        } else {
          setProgress(messages[Math.min(i++, messages.length - 1)]);
        }
      } catch (e) {
        stopPolling();
        setError(String(e));
        setStep("error");
      }
    }, 1200);
  }, [stopPolling]);

  const handleSubmit = async () => {
    if (!beat || !vocal) return;
    setError("");
    setStep("processing");
    setProgress("Uploading…");
    try {
      const { job_id } = await uploadFiles(beat, vocal, preset);
      startPolling(job_id);
    } catch (e) {
      setError(String(e));
      setStep("error");
    }
  };

  const reset = () => {
    stopPolling();
    setBeat(null);
    setVocal(null);
    setJob(null);
    setError("");
    setStep("upload");
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white font-sans">
      {/* Header */}
      <header className="border-b border-white/10 px-6 py-4 flex items-center gap-3">
        <div className="w-7 h-7 rounded-md bg-lime-400 flex items-center justify-center text-black font-black text-sm">B</div>
        <span className="font-bold text-lg tracking-tight">BeatFlow</span>
        <span className="text-white/30 text-sm ml-1">— AI Vocal Mix Assistant</span>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-12 space-y-10">

        {/* ── Upload step ── */}
        {step === "upload" && (
          <div className="space-y-8">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Raw vocals in.</h1>
              <h1 className="text-3xl font-bold tracking-tight text-lime-400">Polished demo out.</h1>
              <p className="mt-3 text-white/50 text-sm max-w-md">
                Upload your beat and raw vocal. BeatFlow analyses, processes, and delivers a clean demo mix in seconds.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <FileDropzone label="Beat" file={beat} onChange={setBeat} />
              <FileDropzone label="Raw Vocal" file={vocal} onChange={setVocal} />
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs text-white/50 uppercase tracking-wider">Genre preset:</span>
              {PRESETS.map((p) => (
                <button
                  key={p.value}
                  onClick={() => setPreset(p.value)}
                  className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-colors
                    ${preset === p.value ? "bg-lime-400 text-black" : "bg-white/10 text-white/60 hover:bg-white/20"}`}
                >
                  {p.label}
                </button>
              ))}
            </div>

            <button
              onClick={handleSubmit}
              disabled={!beat || !vocal}
              className="w-full py-4 rounded-xl bg-lime-400 text-black font-bold text-sm uppercase tracking-wider
                disabled:opacity-30 disabled:cursor-not-allowed hover:bg-lime-300 transition-colors"
            >
              Mix My Vocals →
            </button>
          </div>
        )}

        {/* ── Processing step ── */}
        {step === "processing" && (
          <div className="flex flex-col items-center justify-center py-24 space-y-6">
            <div className="flex gap-1">
              {[0, 1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="w-1 rounded-full bg-lime-400 animate-bounce"
                  style={{ height: 32 + Math.sin(i) * 16, animationDelay: `${i * 100}ms` }}
                />
              ))}
            </div>
            <p className="text-white/70 text-sm">{progress}</p>
          </div>
        )}

        {/* ── Done step ── */}
        {step === "done" && job && (
          <div className="space-y-8">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">Demo Mix Ready.</h2>
                <p className="text-white/50 text-sm mt-1">Your vocal has been processed and balanced.</p>
              </div>
              <button onClick={reset} className="text-xs text-white/40 hover:text-white/70 underline">
                Start over
              </button>
            </div>

            <ReferenceUpload jobId={job.id} onUploaded={() => {}} />

            {/* A/B comparison */}
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider">A/B Comparison</h3>
              <WaveformPlayer url={exportUrl(job.id, "vocal_raw.wav")} label="Before — Raw Vocal" color="#94a3b8" />
              <WaveformPlayer url={exportUrl(job.id, "output.wav")} label="After — Processed Mix" color="#a3e635" />
            </div>

            <ProcessingStats job={job} />

            {/* Export */}
            <div className="flex flex-wrap gap-3">
              <a
                href={exportUrl(job.id, "output.wav")}
                download="beatflow_mix.wav"
                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-lime-400 text-black font-bold text-sm hover:bg-lime-300 transition-colors"
              >
                ↓ Download Mix (WAV)
              </a>
              <a
                href={exportUrl(job.id, "beat.wav")}
                download="beat.wav"
                className="flex items-center gap-2 px-5 py-3 rounded-xl bg-white/10 text-white/70 text-sm hover:bg-white/20 transition-colors"
              >
                ↓ Beat (Original)
              </a>
            </div>
          </div>
        )}

        {/* ── Error step ── */}
        {step === "error" && (
          <div className="space-y-4">
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-6">
              <h2 className="font-bold text-red-400 mb-2">Processing Failed</h2>
              <pre className="text-xs text-red-300/70 whitespace-pre-wrap break-all">{error}</pre>
            </div>
            <button onClick={reset} className="text-sm text-white/50 hover:text-white underline">
              Try again
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
