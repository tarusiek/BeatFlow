"use client";
import { useCallback, useState } from "react";

interface Props {
  label: string;
  accept?: string;
  file: File | null;
  onChange: (f: File | null) => void;
}

export default function FileDropzone({ label, accept = "audio/*", file, onChange }: Props) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) onChange(f);
    },
    [onChange],
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 transition-all cursor-pointer
        ${dragging ? "border-lime-400 bg-lime-400/5" : "border-white/20 hover:border-white/40"}
        ${file ? "bg-white/5" : ""}`}
      onClick={() => document.getElementById(`input-${label}`)?.click()}
    >
      <input
        id={`input-${label}`}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
      />
      <div className="text-3xl mb-2">{file ? "🎵" : "📂"}</div>
      <p className="text-sm font-semibold text-white/70 uppercase tracking-wider">{label}</p>
      {file ? (
        <p className="mt-1 text-xs text-lime-400 truncate max-w-full px-4">{file.name}</p>
      ) : (
        <p className="mt-1 text-xs text-white/40">Drop or click to select</p>
      )}
    </div>
  );
}
