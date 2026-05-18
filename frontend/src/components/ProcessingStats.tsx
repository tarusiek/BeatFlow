"use client";
import type { Job } from "@/lib/api";

interface Props {
  job: Job;
}

function Stat({ label, value, unit = "" }: { label: string; value: string | number | null; unit?: string }) {
  return (
    <div className="flex flex-col items-center bg-white/5 rounded-lg px-4 py-3">
      <span className="text-xs text-white/40 uppercase tracking-wider mb-1">{label}</span>
      <span className="text-lg font-bold text-white">
        {value !== null ? `${typeof value === "number" ? value.toFixed(1) : value}${unit}` : "—"}
      </span>
    </div>
  );
}

export default function ProcessingStats({ job }: Props) {
  const params = job.params as Record<string, number | boolean> | null;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider">Processing Results</h3>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Stat label="Raw Vocal" value={job.before_vocal_lufs} unit=" LUFS" />
        <Stat label="Processed Vocal" value={job.after_vocal_lufs} unit=" LUFS" />
        <Stat label="Mix LUFS" value={job.after_mix_lufs} unit=" LUFS" />
        <Stat label="Preset" value={job.preset ?? "default"} />
      </div>

      {params && (
        <details className="bg-white/5 rounded-xl border border-white/10 p-4">
          <summary className="text-xs text-white/50 cursor-pointer hover:text-white/70 uppercase tracking-wider">
            Decision Engine Parameters
          </summary>
          <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
            {Object.entries(params).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-white/50">{k}</span>
                <span className="text-lime-400">
                  {typeof v === "boolean" ? (v ? "✓" : "✗") : typeof v === "number" ? v.toFixed(2) : String(v)}
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
