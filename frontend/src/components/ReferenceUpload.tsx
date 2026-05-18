"use client";
import { useState } from "react";

interface Props {
  jobId: string;
  onUploaded: () => void;
}

export default function ReferenceUpload({ jobId, onUploaded }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("reference", file);
      const r = await fetch(`/api/upload/reference/${jobId}`, {
        method: "POST",
        body: form,
      });
      if (!r.ok) throw new Error(await r.text());
      setDone(true);
      onUploaded();
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading(false);
    }
  };

  if (done) {
    return (
      <div className="flex items-center gap-2 text-sm text-lime-400">
        <span>✓</span>
        <span>Reference track uploaded — mix will be adjusted to match it.</span>
      </div>
    );
  }

  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4 space-y-3">
      <div>
        <h4 className="text-sm font-semibold text-white/70">Match Reference Track</h4>
        <p className="text-xs text-white/40 mt-0.5">
          Upload a professionally mixed song as a reference. BeatFlow will nudge the processing chain to match its characteristics.
        </p>
      </div>
      <div className="flex gap-3 items-center">
        <label className="flex-1 cursor-pointer">
          <input
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <div className="border border-dashed border-white/20 rounded-lg px-4 py-2 text-xs text-white/50 hover:border-white/40 transition-colors truncate">
            {file ? file.name : "Select reference track…"}
          </div>
        </label>
        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="px-4 py-2 rounded-lg bg-lavender-400/20 hover:bg-purple-500/20 disabled:opacity-40
            text-purple-300 text-xs font-semibold transition-colors border border-purple-500/30"
        >
          {uploading ? "Uploading…" : "Apply"}
        </button>
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}
