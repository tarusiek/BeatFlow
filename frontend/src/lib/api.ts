// In dev, Next.js rewrites /api/* → http://localhost:8000/api/*
// In prod, set NEXT_PUBLIC_API_URL to the backend origin.
const BASE = typeof window !== "undefined"
  ? (process.env.NEXT_PUBLIC_API_URL ?? "")
  : (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000");

export interface Job {
  id: string;
  status: "pending" | "processing" | "done" | "error";
  preset: string | null;
  before_vocal_lufs: number | null;
  after_vocal_lufs: number | null;
  before_mix_lufs: number | null;
  after_mix_lufs: number | null;
  params: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export async function uploadFiles(
  beat: File,
  vocal: File,
  preset: string,
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("beat", beat);
  form.append("vocal", vocal);
  form.append("preset", preset);

  const r = await fetch(`${BASE}/api/upload`, { method: "POST", body: form });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const r = await fetch(`${BASE}/api/jobs/${jobId}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function exportUrl(jobId: string, filename: string): string {
  return `${BASE}/api/export/${jobId}/${filename}`;
}
