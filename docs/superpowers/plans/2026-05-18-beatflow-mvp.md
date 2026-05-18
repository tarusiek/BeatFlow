# BeatFlow — AI Music Production Automation: Full Technical Plan + MVP Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally-runnable Python MVP that automatically cleans, EQs, compresses, and balances a raw vocal against a beat, then exports a processed stereo mix — proving the core value proposition before any web infrastructure is built.

**Architecture:** Feature extraction (librosa) → rule-based parameter decision engine → DSP chain (pedalboard + noisereduce) → gain staging (pyloudnorm) → WAV/MP3 export. No GPU, no cloud, no training required for Stage 1.

**Tech Stack:** Python 3.11, librosa, pedalboard (Spotify), pyloudnorm, noisereduce, soundfile, numpy, scipy, ffmpeg-python, pytest

---

## Part I: Analysis & Architecture

---

### 1. What Is Technically Realistic Right Now

These things work today, with off-the-shelf Python libraries, no custom model training, and no GPU:

**Audio feature extraction**
- Integrated loudness (LUFS) — `pyloudnorm`, mature and accurate
- Dynamic range (crest factor, short-term LUFS variance) — `librosa`
- Noise floor estimation (percentile RMS across frames) — `librosa`
- Spectral centroid, rolloff, contrast — `librosa`
- Sibilance level (energy ratio in 5–10 kHz band) — `librosa` FFT
- Tempo / BPM — `librosa.beat.beat_track`
- Transient density (onset detection) — `librosa.onset.onset_detect`
- Clipping detection (samples at ±1.0) — `numpy`

**DSP processing (no training required)**
- Highpass filter, lowpass filter, shelf EQ, peak EQ — `pedalboard`
- Noise gate — `pedalboard`
- Compressor (fixed-ratio, feed-forward) — `pedalboard`
- Limiter — `pedalboard`
- Reverb (Freeverb algorithm) — `pedalboard`
- Delay — `pedalboard`
- Saturation / distortion — `pedalboard`
- Pitch shift — `pedalboard`
- Noise reduction (spectral subtraction / Wiener filter) — `noisereduce`
- Loudness normalization to LUFS target — `pyloudnorm`
- Stereo panning (multiply channels) — `numpy`

**Automatic decisions (rule-based, defensible)**
- "If noise floor > -50 dBFS, enable noise gate at threshold + 6 dB" — deterministic
- "If dynamic range > 20 dB, ratio = 4:1; else ratio = 2.5:1" — deterministic
- "If sibilance ratio > 1.5, apply de-esser at 7.5 kHz" — deterministic
- "Normalize vocal to -20 LUFS, beat to -16 LUFS before mixing" — deterministic
- "Export final mix to -14 LUFS integrated, -1 dBTP true peak" — industry standard

**Source separation**
- Separate beat components (kick, bass, melody) from a mixed instrumental — `demucs` (CPU-capable, slower)
- Separate vocal from a mixed track if user only has a full song — `demucs`

---

### 2. What Is Unrealistic or Very Hard

Be honest about these limitations before building. Do not pretend they are solved.

**"World-class automatic mixing"**
Not achievable. Professional mixing involves subjective taste decisions that are highly genre- and context-specific. Rule-based systems plateau at "decent starting point," not "radio-ready master."

**Creative EQ decisions without reference**
Deciding whether a vocal should be bright, dark, or mid-forward requires knowing the genre, the artist's intent, and how it sits in a specific beat. A system with no reference input will make statistically average decisions that may be wrong for any specific track.

**Automatic pitch correction (AutoTune behavior)**
Correcting pitchy vocals requires detecting note targets (the scale/key), then warping pitch. This is hard to do without:
- Knowing the key of the song
- Having a pitch detection model that handles vibrato/melisma correctly
- Avoiding artifacts on long notes
`librosa.yin` / `crepe` (neural pitch detection) can estimate pitch, but applying smooth correction without audible artifacts is non-trivial. Exclude from MVP.

**Intelligent reverb/delay tail selection**
Choosing reverb size, pre-delay, and decay that sounds "right" for a genre is a taste decision. A 50ms pre-delay and 1.5s decay sounds good on hip-hop; a plate reverb sounds right on R&B. There is no DSP measurement that tells you which to choose. Use tempo-synchronized presets as a workaround.

**Mix translation (sounds good on speakers AND earbuds)**
Achieving translation across playback systems requires trained ears and reference monitors. No algorithm reliably predicts this without a perceptual model trained on human ratings.

**"Like [Famous Producer]" style cloning**
Illegal as a product claim, technically underdefined ("what exactly is Metro Boomin's mixing chain?"), and misleading. Do not build this.

**Beat/vocal balance that feels "right" for all tempos and genres**
The correct vocal loudness relative to a beat varies by genre (trap vs. R&B vs. pop), tempo, and vocal style (whisper vs. belting). A fixed LUFS differential is a starting point, not a solution.

---

### 3. What Requires Custom ML or Large Datasets

These are post-MVP, post-revenue concerns:

| Feature | Why ML is needed | Dataset required |
|---|---|---|
| "Good vs. bad vocal" classification | Subjective — requires human ratings | 10k+ rated vocal pairs |
| Genre-aware parameter selection | Genre classification from audio | 50k+ tagged tracks |
| Mixing style transfer | Map from one mix style to another | Paired dry/wet stems per style |
| Masking artifact detection | Detect when vocal disappears in mix | Annotated spectral masks |
| Intelligent reverb selection | Match reverb to genre/tempo/vocal style | Rated reverb A/B pairs |
| Neural pitch correction | Smooth intonation fixing | Clean + pitchy vocal pairs |
| Vocal enhancement (upsampling) | Improve lossy/low-quality vocals | Paired lo-fi / hi-fi vocals |

None of these are needed for MVP. All of them require significant data collection and labeling effort.

---

### 4. MVP Definition

**The MVP proves one thing:**
> Given a raw vocal WAV and a beat WAV, the system produces a processed mix where the vocal is cleaner, more present, and better balanced than the raw mix.

**MVP constraints:**
- Runs entirely locally on Windows (no cloud, no GPU required)
- Single Python script, no web server
- Processes audio in under 60 seconds for a 4-minute track on a modern CPU
- No model training required
- Total cost: $0 (all libraries are free/open-source)
- One developer can implement Stage 1 in 2–5 days

**What the MVP does NOT include:**
- Web UI
- Reference track comparison
- Pitch correction
- Stem separation (demucs) — optional add-on
- Real-time processing
- Plugin chain UI

---

### 5. MVP Feature Set

**Included (Stage 1):**
- [x] Load beat.wav and vocal.wav (stereo or mono, any sample rate)
- [x] Resample everything to 44100 Hz
- [x] Analyze vocal features (noise floor, dynamic range, sibilance, spectral centroid)
- [x] Apply vocal cleanup chain:
  - Noise reduction (spectral subtraction)
  - Highpass filter (80 Hz)
  - Noise gate (adaptive threshold)
  - Parametric EQ (mud cut at 300 Hz, presence boost at 3.5 kHz, air shelf at 12 kHz)
  - Compressor (ratio/threshold decided from dynamic range analysis)
  - De-esser (manual spectral gain reduction if sibilance detected)
- [x] Normalize vocal to -20 LUFS
- [x] Normalize beat to -16 LUFS
- [x] Sum to stereo mix
- [x] Apply master limiter (-1 dBTP true peak)
- [x] Normalize mix to -14 LUFS integrated
- [x] Export as 24-bit WAV
- [x] Print before/after LUFS and processing decisions to console

**Included (Stage 3, web layer):**
- [ ] FastAPI backend with file upload endpoint
- [ ] Celery task queue with Redis
- [ ] Next.js frontend with drag-and-drop upload
- [ ] Waveform preview (wavesurfer.js)
- [ ] A/B playback (before vs. after)
- [ ] Track mute/solo/volume controls
- [ ] Export button (WAV + MP3)

**Deferred (Stage 5+):**
- [ ] Reference track analysis and matching
- [ ] Backing vocal / adlib processing
- [ ] Auto pitch correction
- [ ] Genre detection
- [ ] LLM-based parameter decisions
- [ ] User rating feedback loop

---

### 6. Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Stage 1: Local CLI                     │
│                                                             │
│  beat.wav + vocal.wav                                       │
│       │                                                     │
│       ▼                                                     │
│  [features.py]  ──→  extract loudness, noise, sibilance    │
│       │                                                     │
│       ▼                                                     │
│  [decisions.py] ──→  rule-based parameter selection        │
│       │                                                     │
│       ▼                                                     │
│  [vocal_chain.py] ─→ noise reduce → EQ → compress → deess  │
│       │                                                     │
│       ▼                                                     │
│  [mixer.py]     ──→  LUFS normalize + sum stereo           │
│       │                                                     │
│       ▼                                                     │
│  [master.py]    ──→  limit → normalize to -14 LUFS         │
│       │                                                     │
│       ▼                                                     │
│  processed_mix.wav                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Stage 3: Web Application                  │
│                                                             │
│  Browser (Next.js)                                          │
│    │  upload files (multipart)                             │
│    ▼                                                        │
│  FastAPI (Python)                                           │
│    │  enqueue job                                           │
│    ▼                                                        │
│  Redis → Celery Worker                                      │
│    │  runs audio pipeline (same code as Stage 1)           │
│    ▼                                                        │
│  Local filesystem / S3                                      │
│    │  processed files                                       │
│    ▼                                                        │
│  Browser (wavesurfer.js A/B player)                         │
└─────────────────────────────────────────────────────────────┘
```

**Database:** SQLite (dev) / PostgreSQL (prod)
- Stores: job_id, status, input_file_paths, output_file_path, processing_params, before_lufs, after_lufs, created_at

**File storage:** Local `uploads/` and `outputs/` directories (dev), S3-compatible (prod)

**Job queue:** Celery + Redis (allows async processing without blocking the HTTP request)

**Deployment options:**
- Dev: `uvicorn` + `celery -A workers.tasks worker` locally on Windows
- Prod cheap: single DigitalOcean droplet (4GB RAM), Docker Compose
- Prod GPU: RunPod / Lambda Labs for demucs-heavy workloads

---

### 7. Recommended Technologies

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.11 | Best audio DSP ecosystem |
| Feature extraction | librosa 0.10 | Industry standard for audio analysis |
| DSP chain | pedalboard 0.9 (Spotify) | C++-backed, real-time capable, rich plugin set |
| Noise reduction | noisereduce 3.0 | Simple API, Wiener filter approach |
| Loudness | pyloudnorm 0.1 | Accurate ITU-R BS.1770 LUFS measurement |
| File I/O | soundfile 0.12 | Fast, supports 24/32-bit WAV |
| Format conversion | ffmpeg-python | WAV → MP3, format normalization |
| Numerics | numpy, scipy | Array operations, signal processing |
| Backend API | FastAPI | Async, type-safe, auto-docs |
| Task queue | Celery + Redis | Async audio processing jobs |
| ORM | SQLAlchemy 2.0 + SQLite | Lightweight, no server for dev |
| Frontend | Next.js 14 (App Router) | React + SSR, easy deployment |
| Waveform UI | wavesurfer.js 7 | Best waveform visualization in browser |
| Styling | Tailwind CSS | Fast development |
| Testing | pytest + pytest-cov | Standard Python testing |
| Source separation | demucs (optional) | Separate stems from mixed audio |
| Windows dev | WSL2 or native Python | Native Python works fine for audio |

**Do NOT use:**
- Real-time JACK/ASIO audio (not needed for file processing)
- PyTorch from scratch (unnecessary for rule-based MVP)
- Heavy ML frameworks (ONNX only if you add a pre-trained model later)
- Celery Beat (not needed for MVP)

---

### 8. AI Layer Design (Honest, Non-Handwavy)

The "AI" in the MVP is an **interpretable rule-based decision engine** — not a black-box model. This is intentional.

**Step 1: Feature Extraction**

For each input file, extract these measurements:

```
Vocal features:
  - integrated_lufs         (pyloudnorm ITU-R BS.1770-4)
  - peak_true_peak_dbfs     (max sample value)
  - noise_floor_rms         (10th percentile of RMS frames)
  - dynamic_range_db        (95th percentile RMS - 10th percentile RMS, in dB)
  - sibilance_ratio         (mean energy 6–10kHz / mean energy 2–6kHz)
  - spectral_centroid_hz    (brightness measure)
  - zcr_mean                (zero crossing rate — noise/roughness indicator)
  - clipping_fraction       (fraction of samples at ±0.99+)

Beat features:
  - integrated_lufs
  - bpm                     (librosa.beat.beat_track)
  - low_end_energy_ratio    (energy below 200Hz / total energy)
  - spectral_centroid_hz
```

**Step 2: Decision Engine**

Map features to processing parameters using explicit rules:

```python
# Noise gate threshold: sit just above measured noise floor
noise_gate_threshold_db = noise_floor_db + 6 dB  # clamp to [-70, -30]

# Compression ratio: scale with dynamic range
if dynamic_range_db > 22:    ratio = 4.0,  threshold = -20
elif dynamic_range_db > 15:  ratio = 3.0,  threshold = -20
else:                        ratio = 2.0,  threshold = -24

# Attack/release: fast attack on dense transients
if zcr_mean > 0.15:     attack_ms = 5,   release_ms = 80
else:                   attack_ms = 15,  release_ms = 150

# De-esser: enable when sibilance ratio exceeds 1.2
if sibilance_ratio > 1.5:   deess_threshold = -16, deess_freq = 7000
elif sibilance_ratio > 1.2: deess_threshold = -22, deess_freq = 8000
else:                       deess_enabled = False

# EQ: boost presence if vocal is spectrally dark
if spectral_centroid_hz < 2000:  presence_gain = +3 dB at 3500 Hz
elif spectral_centroid_hz < 3000: presence_gain = +2 dB at 4000 Hz
else:                             presence_gain = +1 dB at 5000 Hz

# Beat/vocal balance
vocal_target_lufs = -20.0
beat_target_lufs  = -16.0
```

**Step 3: Processing Chain (in order)**

```
1. Noise reduction  (noisereduce, uses first 0.3s as noise profile)
2. Highpass filter  (80 Hz, 12 dB/oct)
3. Noise gate       (threshold from step 2)
4. Low-mid cut      (-2 dB at 300 Hz, Q=1.0)
5. Presence boost   (from decision engine)
6. High shelf       (+1.5 dB at 12 kHz)
7. Compressor       (threshold/ratio/attack/release from decision engine)
8. De-esser         (if enabled: spectral gain reduction at 6–10 kHz)
9. Gain to -20 LUFS
```

**Step 4: Mix Assembly**

```
1. Normalize beat to -16 LUFS
2. Normalize vocal to -20 LUFS
3. Sum stereo
4. Apply limiting at -1 dBTP
5. Normalize sum to -14 LUFS
6. Final safety limiter pass
```

**Step 5: Evaluation (automated)**

After processing, measure and log:
- Before/after LUFS (vocal, beat, mix)
- Before/after dynamic range
- Before/after sibilance ratio
- True peak of output (must be < -1.0 dBTP)
- Clipping flag (if any sample ≥ 0.99 after master)

**LLM Layer (Stage 6+)**

When rule-based decisions plateau, pass the feature vector and current output evaluation to Claude:

```python
prompt = f"""
Vocal analysis:
  LUFS: {integrated_lufs:.1f}
  Dynamic range: {dynamic_range_db:.1f} dB
  Sibilance ratio: {sibilance_ratio:.2f}
  Spectral centroid: {spectral_centroid_hz:.0f} Hz

Current processing chain applied:
  {current_chain_description}

Output quality:
  Output LUFS: {output_lufs:.1f}
  Sibilance after: {after_sibilance:.2f}
  Dynamic range after: {after_dr:.1f} dB

Suggest adjustments to the processing chain parameters.
Return as JSON with keys: comp_ratio, comp_threshold_db, deess_threshold_db,
presence_gain_db, presence_freq_hz. Only change values that would improve the mix.
"""
```

The LLM acts as a parameter advisor, not a raw audio processor. This is architecturally safe and cheap.

---

### 9. Data & Testing Approach

**Test file organization:**

```
tests/
└── fixtures/
    ├── vocals/
    │   ├── good/         # Clean, well-recorded vocals (label: good)
    │   ├── bad/          # Noisy, harsh, poorly recorded (label: bad)
    │   └── neutral/      # Average recordings
    ├── beats/
    │   ├── hip_hop/
    │   ├── rnb/
    │   └── pop/
    ├── references/       # Full mixed songs as reference
    └── expected_outputs/ # Pre-approved processed outputs for regression
```

**Metadata to store (SQLite or JSON sidecar):**

```json
{
  "file": "vocals/bad/noisy_vocal_01.wav",
  "label": "bad",
  "source": "original_recording",
  "issues": ["noise_floor", "harsh_sibilance", "low_presence"],
  "noise_floor_db": -42.3,
  "dynamic_range_db": 18.5,
  "sibilance_ratio": 2.1,
  "integrated_lufs": -18.2,
  "notes": "Recorded in a bedroom, audible room tone, slightly clipped peaks"
}
```

**Useful evaluation metrics:**

| Metric | What it measures | How to compute |
|---|---|---|
| LUFS before/after | Loudness change | pyloudnorm |
| Dynamic range before/after | Compression effect | percentile RMS ratio |
| Sibilance ratio before/after | De-esser effectiveness | librosa FFT |
| Noise floor before/after | Noise reduction effectiveness | percentile RMS |
| True peak | Clipping safety | pyloudnorm |
| PESQ or VISQOL | Perceptual speech quality | pesq library (optional) |

**Misleading metrics:**
- SNR (signal-to-noise ratio) — depends on signal level, not quality
- LUFS increase alone — loud ≠ better
- Spectral flatness — processed doesn't mean flat
- Waveform similarity (MSE) — different good mixes have different waveforms

**Manual rating protocol:**

1. Listen blind (A/B with labels hidden)
2. Rate on 3 axes: Clarity (1–5), Presence (1–5), Harshness (1–5, lower = better)
3. Record in `tests/fixtures/ratings.csv`: `file, rater, clarity, presence, harshness, notes`
4. After 10+ ratings per file, average scores become ground truth for regression

**Regression test approach:**

After approving an output, store it in `tests/fixtures/expected_outputs/`. A regression test:
1. Processes the same input with current pipeline
2. Compares LUFS, dynamic range, sibilance of output vs. stored expected output
3. Fails if any metric differs by more than a tolerance threshold

---

### 10. Project Folder Structure

```
beatflow/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── upload.py       # POST /upload
│   │   │   ├── jobs.py         # GET /jobs/{id}, PATCH /jobs/{id}
│   │   │   └── export.py       # GET /export/{job_id}
│   │   └── schemas.py          # Pydantic request/response models
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── features.py         # Feature extraction (librosa)
│   │   ├── decisions.py        # Rule-based parameter selection
│   │   ├── vocal_chain.py      # Full vocal processing chain
│   │   ├── deesser.py          # De-esser implementation
│   │   ├── mixer.py            # Gain staging and stem summing
│   │   └── master.py           # Master limiter and final normalization
│   ├── workers/
│   │   ├── __init__.py
│   │   └── tasks.py            # Celery tasks wrapping the audio pipeline
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py           # SQLAlchemy models
│   │   └── session.py          # Database session factory
│   ├── config.py               # Settings (env vars, paths, targets)
│   └── main.py                 # FastAPI application entry point
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx        # Main mixing page
│   │   │   └── api/            # Next.js API routes (proxies to FastAPI)
│   │   ├── components/
│   │   │   ├── TrackUploader.tsx
│   │   │   ├── WaveformViewer.tsx
│   │   │   ├── TrackStrip.tsx  # Mute/solo/vol/pan per track
│   │   │   ├── ProcessingPanel.tsx
│   │   │   └── ABPlayer.tsx    # Before/after comparison
│   │   └── lib/
│   │       ├── api.ts          # Typed fetch client
│   │       └── audio.ts        # Web Audio API helpers
│   ├── public/
│   ├── package.json
│   └── tsconfig.json
├── scripts/
│   └── prototype.py            # Stage 1 offline script (standalone)
├── tests/
│   ├── audio/
│   │   ├── test_features.py
│   │   ├── test_decisions.py
│   │   ├── test_vocal_chain.py
│   │   ├── test_deesser.py
│   │   ├── test_mixer.py
│   │   └── test_master.py
│   ├── api/
│   │   ├── test_upload.py
│   │   └── test_jobs.py
│   └── fixtures/
│       ├── vocals/
│       │   ├── good/
│       │   ├── bad/
│       │   └── neutral/
│       ├── beats/
│       ├── references/
│       ├── expected_outputs/
│       └── ratings.csv
├── docs/
│   ├── architecture.md
│   ├── audio_processing.md
│   ├── ai_decisions.md
│   └── superpowers/
│       └── plans/
│           └── 2026-05-18-beatflow-mvp.md   ← this file
├── configs/
│   ├── processing_presets.yaml  # Named presets: hip_hop, rnb, pop
│   └── feature_targets.yaml     # Target ranges for "good" audio
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

### 11. Development Roadmap

#### Stage 0 — Research & Constraints (Day 1)
- Install and smoke-test all libraries locally
- Download a beat WAV and vocal WAV for testing
- Verify pedalboard runs on Windows without issues
- Measure a test vocal with pyloudnorm to confirm LUFS values are sane
- **Deliverable:** Working Python environment, 2 test WAV files

#### Stage 1 — Offline Python Prototype (Days 2–5)
- Implement feature extraction, decision engine, vocal chain, mixer, master
- Build `scripts/prototype.py` as the entry point
- **Deliverable:** `python scripts/prototype.py beat.wav vocal.wav output.wav` works

#### Stage 2 — Test Suite & Evaluation Dataset (Days 5–7)
- Write pytest tests for all audio modules with synthetic inputs
- Collect 5–10 real vocal samples in `tests/fixtures/`
- Measure before/after and rate manually
- **Deliverable:** `pytest` passes, evaluation spreadsheet

#### Stage 3 — FastAPI Backend (Days 8–12)
- Wrap prototype in FastAPI
- Add SQLite DB for job tracking
- Add Celery + Redis for async processing
- **Deliverable:** `POST /upload` → job queued → `GET /jobs/{id}` returns status

#### Stage 4 — Next.js Frontend + A/B Player (Days 13–18)
- Drag-and-drop upload
- Waveform viewer with wavesurfer.js
- A/B comparison player (before / after)
- Export button
- **Deliverable:** Full web UI working end-to-end on localhost

#### Stage 5 — Reference Track Analysis (Days 19–24)
- Upload reference song → extract its features
- Adjust vocal chain targets to match reference
- **Deliverable:** "Match reference" button works

#### Stage 6 — Smarter Decision Engine (Days 25–35)
- Optional: Claude API integration for parameter suggestions
- Add genre detection (simple classifier or rule-based from spectral features)
- Add preset library (hip_hop, rnb, pop)
- **Deliverable:** Genre-aware processing, LLM parameter suggestions in logs

#### Stage 7 — Production Deployment (Days 36–45)
- Docker Compose: FastAPI + Celery + Redis + Nginx
- S3-compatible file storage
- User auth (JWT)
- Rate limiting
- **Deliverable:** Deployed to DigitalOcean, accessible via URL

#### Stage 8 — Future ML (Post-Revenue)
- Collect user ratings on outputs
- Train a small classifier for "good vs. bad chain" given features
- Integrate audio quality models (PESQ, VISQOL)
- Consider fine-tuned model for genre-specific EQ
- **Deliverable:** Data pipeline + first model experiment

---

### 12. First Implementation Step — Exact Description

Build `scripts/prototype.py`. It must:

1. Accept `beat.wav`, `vocal.wav`, and `output.wav` as CLI arguments
2. Load both files, resample to 44100 Hz stereo
3. Extract vocal features (noise floor, dynamic range, sibilance, spectral centroid)
4. Decide processing parameters from features (explicit rules, logged to console)
5. Apply processing chain: noise reduce → HPF → gate → EQ → compress → de-ess → normalize
6. Normalize beat to -16 LUFS, processed vocal to -20 LUFS
7. Sum to stereo mix
8. Limit to -1 dBTP, normalize to -14 LUFS
9. Export as 24-bit WAV
10. Print a before/after report (LUFS, dynamic range, sibilance ratio)

---

### 13. Pseudocode Prototype

```python
# scripts/prototype.py
# Usage: python prototype.py beat.wav vocal.wav output.wav

import sys
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import noisereduce as nr
from pedalboard import (Pedalboard, HighpassFilter, LowShelfFilter,
                        HighShelfFilter, Compressor, NoiseGate, Limiter, Gain)

SAMPLE_RATE = 44100
TARGET_VOCAL_LUFS = -20.0
TARGET_BEAT_LUFS  = -16.0
TARGET_MIX_LUFS   = -14.0
TRUE_PEAK_LIMIT   = -1.0

# ─── I/O ────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load any audio file, resample to SAMPLE_RATE, return (2, N) float32."""
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    return audio.astype(np.float32)

def save_audio(path: str, audio: np.ndarray) -> None:
    """Save (2, N) float32 as 24-bit stereo WAV."""
    sf.write(path, audio.T, SAMPLE_RATE, subtype='PCM_24')

# ─── LOUDNESS ───────────────────────────────────────────────────────────────

def measure_lufs(audio: np.ndarray) -> float:
    """Measure integrated loudness (LUFS). Returns -inf if signal is silent."""
    meter = pyln.Meter(SAMPLE_RATE)
    loudness = meter.integrated_loudness(audio.T)
    return loudness

def normalize_to_lufs(audio: np.ndarray, target_lufs: float) -> np.ndarray:
    """Apply gain to reach target_lufs. Returns unchanged audio if silent."""
    current = measure_lufs(audio)
    if np.isinf(current):
        return audio
    gain_db = target_lufs - current
    gain_linear = 10.0 ** (gain_db / 20.0)
    return audio * gain_linear

# ─── FEATURE EXTRACTION ─────────────────────────────────────────────────────

def extract_features(audio: np.ndarray) -> dict:
    """Extract relevant audio features from a vocal or instrument track."""
    mono = librosa.to_mono(audio)

    # RMS frames for dynamic analysis
    rms_frames = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
    noise_floor_rms = float(np.percentile(rms_frames, 10))
    peak_rms        = float(np.percentile(rms_frames, 95))

    noise_floor_db  = 20.0 * np.log10(noise_floor_rms + 1e-9)
    dynamic_range   = 20.0 * np.log10((peak_rms + 1e-9) / (noise_floor_rms + 1e-9))

    # Sibilance: energy ratio 6–10 kHz vs 2–6 kHz
    stft  = np.abs(librosa.stft(mono, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    sib_mask  = (freqs >= 6000) & (freqs <= 10000)
    pres_mask = (freqs >= 2000) & (freqs <=  6000)
    sib_energy  = float(stft[sib_mask].mean())
    pres_energy = float(stft[pres_mask].mean())
    sibilance_ratio = sib_energy / (pres_energy + 1e-9)

    # Brightness
    centroid = float(librosa.feature.spectral_centroid(y=mono, sr=SAMPLE_RATE).mean())

    # Zero crossing rate (noise roughness indicator)
    zcr = float(librosa.feature.zero_crossing_rate(mono).mean())

    return {
        'lufs':              measure_lufs(audio),
        'noise_floor_db':    noise_floor_db,
        'dynamic_range_db':  dynamic_range,
        'sibilance_ratio':   sibilance_ratio,
        'spectral_centroid': centroid,
        'zcr_mean':          zcr,
    }

# ─── DECISION ENGINE ────────────────────────────────────────────────────────

def decide_params(features: dict) -> dict:
    """Map extracted features to DSP chain parameters. All rules are explicit."""
    p = {}

    # Noise gate: threshold = noise floor + 6 dB, clamped to [-70, -30]
    p['gate_threshold_db'] = float(np.clip(features['noise_floor_db'] + 6.0, -70.0, -30.0))

    # Compressor: more aggressive for wider dynamic range
    dr = features['dynamic_range_db']
    if dr > 22:
        p['comp_ratio']      = 4.0
        p['comp_threshold']  = -20.0
        p['comp_attack_ms']  = 8.0
        p['comp_release_ms'] = 80.0
    elif dr > 15:
        p['comp_ratio']      = 3.0
        p['comp_threshold']  = -20.0
        p['comp_attack_ms']  = 12.0
        p['comp_release_ms'] = 120.0
    else:
        p['comp_ratio']      = 2.0
        p['comp_threshold']  = -24.0
        p['comp_attack_ms']  = 18.0
        p['comp_release_ms'] = 160.0

    # High-density transients → faster attack
    if features['zcr_mean'] > 0.15:
        p['comp_attack_ms'] = max(4.0, p['comp_attack_ms'] - 6.0)

    # Presence EQ: brighter boost for dark vocals
    centroid = features['spectral_centroid']
    if centroid < 2000:
        p['presence_freq'] = 3500.0
        p['presence_gain'] = 3.0
    elif centroid < 3000:
        p['presence_freq'] = 4000.0
        p['presence_gain'] = 2.0
    else:
        p['presence_freq'] = 5000.0
        p['presence_gain'] = 1.5

    # De-esser
    sr = features['sibilance_ratio']
    if sr > 1.5:
        p['deess_enabled']    = True
        p['deess_freq']       = 7000.0
        p['deess_threshold']  = -16.0
        p['deess_reduction']  = -6.0
    elif sr > 1.2:
        p['deess_enabled']    = True
        p['deess_freq']       = 8000.0
        p['deess_threshold']  = -22.0
        p['deess_reduction']  = -4.0
    else:
        p['deess_enabled'] = False

    return p

# ─── PROCESSING CHAIN ───────────────────────────────────────────────────────

def apply_noise_reduction(audio: np.ndarray) -> np.ndarray:
    """Use first 300ms as noise profile. Apply Wiener-filter noise reduction."""
    noise_sample = audio[:, :int(SAMPLE_RATE * 0.3)]
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, y_noise=noise_sample,
                           prop_decrease=0.75, n_fft=2048)

def apply_deesser(audio: np.ndarray, freq: float, threshold_db: float,
                  reduction_db: float) -> np.ndarray:
    """
    Simple spectral de-esser: detect frames where high-frequency energy
    exceeds threshold, apply gain reduction to those frames only.
    """
    mono     = librosa.to_mono(audio)
    n_fft    = 2048
    hop      = 512
    stft     = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    freqs    = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=n_fft)
    sib_mask = freqs >= freq

    magnitudes = np.abs(stft)
    sib_energy = magnitudes[sib_mask].mean(axis=0)            # (frames,)
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    reduction_linear = 10.0 ** (reduction_db / 20.0)         # < 1.0

    gain_curve = np.ones(sib_energy.shape)
    over_threshold = sib_energy > threshold_linear
    gain_curve[over_threshold] = reduction_linear

    # Smooth gain curve to avoid clicks
    from scipy.ndimage import uniform_filter1d
    gain_curve = uniform_filter1d(gain_curve, size=7)

    # Apply frame-level gain to full audio
    gain_frames = np.interp(
        np.arange(audio.shape[1]),
        np.arange(len(gain_curve)) * hop + n_fft // 2,
        gain_curve
    )
    gain_frames = np.clip(gain_frames, reduction_linear, 1.0)
    return audio * gain_frames[np.newaxis, :]

def apply_vocal_chain(audio: np.ndarray, params: dict) -> np.ndarray:
    """Full vocal processing chain. Returns processed (2, N) float32."""

    # 1. Noise reduction
    audio = apply_noise_reduction(audio)

    # 2. EQ + Gate + Compressor via pedalboard
    chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80.0),
        NoiseGate(
            threshold_db=params['gate_threshold_db'],
            ratio=10.0,
            attack_ms=1.0,
            release_ms=80.0,
        ),
        LowShelfFilter(
            cutoff_frequency_hz=300.0,
            gain_db=-2.0,
        ),
        HighShelfFilter(
            cutoff_frequency_hz=params['presence_freq'],
            gain_db=params['presence_gain'],
        ),
        HighShelfFilter(
            cutoff_frequency_hz=12000.0,
            gain_db=1.5,
        ),
        Compressor(
            threshold_db=params['comp_threshold'],
            ratio=params['comp_ratio'],
            attack_ms=params['comp_attack_ms'],
            release_ms=params['comp_release_ms'],
        ),
    ])
    audio = chain(audio, SAMPLE_RATE)

    # 3. De-esser (spectral, applied separately)
    if params.get('deess_enabled'):
        audio = apply_deesser(
            audio,
            freq=params['deess_freq'],
            threshold_db=params['deess_threshold'],
            reduction_db=params['deess_reduction'],
        )

    return audio

# ─── MASTER CHAIN ───────────────────────────────────────────────────────────

def apply_master_chain(mix: np.ndarray) -> np.ndarray:
    """Limit to -1 dBTP, then normalize to TARGET_MIX_LUFS."""
    limiter = Pedalboard([Limiter(threshold_db=TRUE_PEAK_LIMIT, release_ms=100.0)])
    mix = limiter(mix, SAMPLE_RATE)
    mix = normalize_to_lufs(mix, TARGET_MIX_LUFS)
    mix = limiter(mix, SAMPLE_RATE)   # Second pass safety
    return mix

# ─── MIX ASSEMBLY ───────────────────────────────────────────────────────────

def pad_to_same_length(a: np.ndarray, b: np.ndarray) -> tuple:
    """Zero-pad shorter array to match longer."""
    max_len = max(a.shape[1], b.shape[1])
    a_pad = np.pad(a, ((0, 0), (0, max_len - a.shape[1])))
    b_pad = np.pad(b, ((0, 0), (0, max_len - b.shape[1])))
    return a_pad, b_pad

# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 4:
        print("Usage: python prototype.py beat.wav vocal.wav output.wav")
        sys.exit(1)

    beat_path, vocal_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    print("─── Loading ───────────────────────────────────")
    beat  = load_audio(beat_path)
    vocal = load_audio(vocal_path)
    print(f"  Beat:  {beat.shape[1] / SAMPLE_RATE:.1f}s  LUFS={measure_lufs(beat):.1f}")
    print(f"  Vocal: {vocal.shape[1] / SAMPLE_RATE:.1f}s  LUFS={measure_lufs(vocal):.1f}")

    print("─── Analyzing Vocal ───────────────────────────")
    features = extract_features(vocal)
    for k, v in features.items():
        print(f"  {k:25s}: {v:.3f}")

    print("─── Deciding Parameters ───────────────────────")
    params = decide_params(features)
    for k, v in params.items():
        print(f"  {k:25s}: {v}")

    print("─── Processing Vocal ──────────────────────────")
    vocal_processed = apply_vocal_chain(vocal, params)

    print("─── Balancing Mix ─────────────────────────────")
    beat_balanced  = normalize_to_lufs(beat,            TARGET_BEAT_LUFS)
    vocal_balanced = normalize_to_lufs(vocal_processed, TARGET_VOCAL_LUFS)

    print("─── Summing Mix ───────────────────────────────")
    beat_padded, vocal_padded = pad_to_same_length(beat_balanced, vocal_balanced)
    mix = beat_padded + vocal_padded

    print("─── Mastering ─────────────────────────────────")
    master = apply_master_chain(mix)

    print("─── Exporting ─────────────────────────────────")
    save_audio(output_path, master)

    # Before/after report
    raw_mix, _ = pad_to_same_length(beat, vocal)
    raw_mix_summed = raw_mix + _
    print("\n─── Report ────────────────────────────────────")
    print(f"  Input vocal LUFS:      {features['lufs']:.1f}")
    print(f"  Input dynamic range:   {features['dynamic_range_db']:.1f} dB")
    print(f"  Input sibilance ratio: {features['sibilance_ratio']:.2f}")
    after = extract_features(vocal_processed)
    print(f"  Output vocal LUFS:     {measure_lufs(vocal_processed):.1f}")
    print(f"  Output dynamic range:  {after['dynamic_range_db']:.1f} dB")
    print(f"  Output sibilance:      {after['sibilance_ratio']:.2f}")
    print(f"  Final mix LUFS:        {measure_lufs(master):.1f}")
    print(f"  Saved to:              {output_path}")

if __name__ == "__main__":
    main()
```

---

### 14. Hardest Technical Problems

| Problem | Why It's Hard | MVP Workaround | Long-Term Solution |
|---|---|---|---|
| **Noise profile estimation** | First 300ms may not be noise-only if vocal starts immediately | Use last 300ms if start is loud; flag to user | Let user mark a "noise-only" region |
| **De-esser on melodic content** | Melody in 5–10kHz range will be attenuated by aggressive de-essing | Only reduce by 4–6 dB max, use frame-by-frame detection | Perceptual model that distinguishes sibilance from melody |
| **Beat/vocal LUFS balance** | -4 dB differential is correct for some genres, wrong for others | Use -4 dB as default, expose as adjustable parameter | Genre-aware preset selection |
| **Clipping on loud transients** | After normalization, transient peaks can exceed 0 dBFS | Limiter at -1 dBTP before any normalization, check output | True peak aware normalization with lookahead limiting |
| **Mono compatibility** | Stereo reverb/effects can cancel in mono playback | Keep reverb wet mix ≤ 20%, avoid mid-side processing in MVP | Test through mono summing, adjust wet/dry |
| **Pedalboard LowShelfFilter as presence EQ** | LowShelfFilter is the wrong curve for a midrange presence boost | Use HighShelfFilter with a lower cutoff as a workaround; note: PeakFilter available in pedalboard ≥ 0.9.14 | Upgrade to PeakFilter once confirmed available |
| **Sample rate mismatch** | User uploads 48kHz beat + 44.1kHz vocal | Always resample to 44100 in load_audio() | Support 48kHz pipeline end-to-end |
| **Stereo vs mono vocal** | Processing a mono vocal into a stereo mix needs panning | Stack mono to pseudo-stereo; add slight pre-delay on one channel for width | True mid-side processing with stereo widener |

---

### 15. Legal & Product Risks

**Copyright risks:**
- User uploads a copyrighted beat → you store it → you're processing it: you are likely safe as a processing service (similar to cloud storage / DAW), but do NOT redistribute processed mixes publicly or train on uploaded user audio without explicit consent.
- Reference tracks: users often upload commercial songs → extract features only, do not store the audio longer than the session, add a ToS clause.
- Do NOT use famous producer names (Metro Boomin, Murda Beatz, etc.) in your marketing or product copy. This implies endorsement and creates false association.

**Privacy:**
- Uploaded audio may contain voice recordings → covered by GDPR in EU, various US state laws. Store minimum time needed, implement deletion.
- Do NOT send user audio to third-party APIs (including Claude API) without disclosing this in ToS.

**Model/API terms:**
- pedalboard: Apache 2.0 — no restrictions
- librosa: ISC — no restrictions
- noisereduce: MIT — no restrictions
- pyloudnorm: MIT — no restrictions
- demucs: MIT + CC BY-NC-ND 4.0 for the pretrained weights — the weights are **non-commercial**. Do NOT use demucs in a paid service without reviewing the license. The code is MIT; only the model weights are restricted.
- If you use Claude API for decision suggestions: Anthropic ToS prohibits using outputs to train competing models. You can use Claude to suggest parameters; you cannot train a model on Claude's outputs.

**Marketing claims:**
- Do NOT claim: "Professional quality", "Studio-quality", "Like a real producer", "Sounds like [name]"
- DO claim: "Automatic starting point", "AI-assisted", "Processed mix", "Cleaner vocals"

---

### 16. Product Positioning

**Wrong positioning (avoid):**
> "AI music producer that replaces mixing engineers"

This is overpromising, will get negative reviews when output is imperfect, and invites legal risk.

**Correct positioning (choose one):**

**Option A — Demo Mix Assistant**
> "BeatFlow turns your raw vocal and beat into a polished demo mix in seconds. Not a final master — a clean starting point."

**Option B — Vocal Polish Tool for Rappers & Singers**
> "Upload your voice, get it cleaned, compressed, and balanced. No DAW knowledge required."

**Option C — AI Mix Starting Point**
> "Automatically apply a professional-grade vocal chain and gain balance. Then tweak from there."

**Recommendation:** Option B is the most specific, most honest, and most compelling for the target market (independent rappers, home studio singers, content creators). It makes one specific promise (vocal sounds better) rather than a vague "AI producer" claim.

---

---

## Part II: Stage 1 Implementation Tasks

---

### Task 0: Environment Setup

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `scripts/prototype.py` (empty, add `if __name__ == "__main__": pass`)
- Create: `backend/audio/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/audio/__init__.py`
- Create: `tests/fixtures/` (directory)

- [ ] **Step 1: Create project directories**

```powershell
# Run from D:\APKI\BeatFlow
New-Item -ItemType Directory -Force backend/audio
New-Item -ItemType Directory -Force backend/api/routes
New-Item -ItemType Directory -Force scripts
New-Item -ItemType Directory -Force tests/audio
New-Item -ItemType Directory -Force tests/fixtures/vocals/good
New-Item -ItemType Directory -Force tests/fixtures/vocals/bad
New-Item -ItemType Directory -Force tests/fixtures/beats
New-Item -ItemType Directory -Force configs
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "beatflow"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "librosa>=0.10.2",
    "soundfile>=0.12.1",
    "noisereduce>=3.0.3",
    "pedalboard>=0.9.14",
    "pyloudnorm>=0.1.1",
    "numpy>=1.26.0",
    "scipy>=1.13.0",
    "ffmpeg-python>=0.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-cov>=5.0.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create virtual environment and install**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

- [ ] **Step 4: Verify installation**

```powershell
python -c "import librosa, soundfile, noisereduce, pedalboard, pyloudnorm; print('OK')"
```

Expected output: `OK`

- [ ] **Step 5: Create empty init files**

```powershell
@("backend/__init__.py", "backend/audio/__init__.py", "tests/__init__.py", "tests/audio/__init__.py", "scripts/__init__.py") | ForEach-Object { New-Item -Force $_ }
```

- [ ] **Step 6: Commit**

```powershell
git init
git add pyproject.toml requirements.txt backend tests scripts
git commit -m "chore: initialize beatflow project structure"
```

---

### Task 1: Loudness Measurement Module

**Files:**
- Create: `backend/audio/loudness.py`
- Test: `tests/audio/test_loudness.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audio/test_loudness.py`:

```python
import numpy as np
import pytest
from backend.audio.loudness import measure_lufs, normalize_to_lufs

SAMPLE_RATE = 44100


def _sine_stereo(freq_hz: float = 440.0, duration_s: float = 3.0,
                 amplitude: float = 0.5) -> np.ndarray:
    """Generate a stereo sine wave as (2, N) float32."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    return np.stack([mono, mono])


def test_measure_lufs_returns_float():
    audio = _sine_stereo()
    result = measure_lufs(audio, SAMPLE_RATE)
    assert isinstance(result, float)


def test_measure_lufs_silent_signal():
    audio = np.zeros((2, SAMPLE_RATE * 3), dtype=np.float32)
    result = measure_lufs(audio, SAMPLE_RATE)
    assert np.isinf(result) or result < -70.0


def test_measure_lufs_reasonable_range_for_half_amplitude_sine():
    # A 440 Hz sine at 0.5 amplitude should be roughly -23 LUFS ±5
    audio = _sine_stereo(amplitude=0.5, duration_s=5.0)
    result = measure_lufs(audio, SAMPLE_RATE)
    assert -30.0 < result < -15.0, f"Unexpected LUFS: {result}"


def test_normalize_to_lufs_reaches_target():
    audio = _sine_stereo(amplitude=0.1, duration_s=5.0)
    target = -20.0
    normalized = normalize_to_lufs(audio, SAMPLE_RATE, target)
    result = measure_lufs(normalized, SAMPLE_RATE)
    assert abs(result - target) < 1.0, f"Expected ~{target} LUFS, got {result:.2f}"


def test_normalize_to_lufs_silent_returns_unchanged():
    audio = np.zeros((2, SAMPLE_RATE * 3), dtype=np.float32)
    result = normalize_to_lufs(audio, SAMPLE_RATE, -20.0)
    assert np.allclose(result, audio)


def test_normalize_to_lufs_preserves_shape():
    audio = _sine_stereo()
    result = normalize_to_lufs(audio, SAMPLE_RATE, -18.0)
    assert result.shape == audio.shape
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/audio/test_loudness.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.audio.loudness'`

- [ ] **Step 3: Implement `backend/audio/loudness.py`**

```python
import numpy as np
import pyloudnorm as pyln


def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """
    Measure integrated loudness (ITU-R BS.1770-4).
    audio: (2, N) or (N,) float32
    Returns LUFS as float; -inf if signal too quiet to measure.
    """
    meter = pyln.Meter(sample_rate)
    if audio.ndim == 2:
        loudness = meter.integrated_loudness(audio.T)
    else:
        loudness = meter.integrated_loudness(audio)
    return float(loudness)


def normalize_to_lufs(audio: np.ndarray, sample_rate: int,
                      target_lufs: float) -> np.ndarray:
    """Apply linear gain to reach target_lufs. Returns audio unchanged if silent."""
    current = measure_lufs(audio, sample_rate)
    if np.isinf(current):
        return audio
    gain_db = target_lufs - current
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (audio * gain_linear).astype(np.float32)
```

- [ ] **Step 4: Run tests and verify they pass**

```powershell
pytest tests/audio/test_loudness.py -v
```

Expected: All 6 tests `PASSED`

- [ ] **Step 5: Commit**

```powershell
git add backend/audio/loudness.py tests/audio/test_loudness.py
git commit -m "feat(audio): add loudness measurement and LUFS normalization"
```

---

### Task 2: Feature Extraction Module

**Files:**
- Create: `backend/audio/features.py`
- Test: `tests/audio/test_features.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audio/test_features.py`:

```python
import numpy as np
import pytest
from backend.audio.features import extract_features

SAMPLE_RATE = 44100


def _white_noise_stereo(duration_s: float = 3.0, amplitude: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(42)
    mono = (amplitude * rng.standard_normal(int(SAMPLE_RATE * duration_s))).astype(np.float32)
    mono = np.clip(mono, -1.0, 1.0)
    return np.stack([mono, mono])


def _sine_with_noise(duration_s: float = 3.0) -> np.ndarray:
    """A 440 Hz sine with a low-level noise floor."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rng = np.random.default_rng(7)
    noise = 0.005 * rng.standard_normal(len(t)).astype(np.float32)
    mono = signal + noise
    return np.stack([mono, mono])


def test_extract_features_returns_all_keys():
    audio = _sine_with_noise()
    features = extract_features(audio, SAMPLE_RATE)
    expected_keys = {
        'lufs', 'noise_floor_db', 'dynamic_range_db',
        'sibilance_ratio', 'spectral_centroid', 'zcr_mean',
    }
    assert expected_keys.issubset(features.keys())


def test_extract_features_all_values_are_floats():
    audio = _sine_with_noise()
    features = extract_features(audio, SAMPLE_RATE)
    for k, v in features.items():
        assert isinstance(v, float), f"{k} is not float: {type(v)}"


def test_dynamic_range_positive():
    audio = _sine_with_noise()
    features = extract_features(audio, SAMPLE_RATE)
    assert features['dynamic_range_db'] > 0.0


def test_noise_floor_below_peak():
    audio = _sine_with_noise()
    features = extract_features(audio, SAMPLE_RATE)
    assert features['noise_floor_db'] < features['lufs']


def test_sibilance_ratio_positive():
    audio = _white_noise_stereo()
    features = extract_features(audio, SAMPLE_RATE)
    # White noise has roughly equal energy across bands, ratio near 1.0
    assert 0.5 < features['sibilance_ratio'] < 3.0


def test_high_frequency_signal_has_high_sibilance():
    """A 8kHz sine should have very high sibilance ratio."""
    t = np.linspace(0, 3.0, int(SAMPLE_RATE * 3.0), endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32)
    audio = np.stack([mono, mono])
    features = extract_features(audio, SAMPLE_RATE)
    assert features['sibilance_ratio'] > 2.0


def test_spectral_centroid_higher_for_bright_signal():
    t = np.linspace(0, 3.0, int(SAMPLE_RATE * 3.0), endpoint=False)
    low_mono  = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    high_mono = (0.3 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)
    low_audio  = np.stack([low_mono,  low_mono])
    high_audio = np.stack([high_mono, high_mono])
    low_feat  = extract_features(low_audio,  SAMPLE_RATE)
    high_feat = extract_features(high_audio, SAMPLE_RATE)
    assert high_feat['spectral_centroid'] > low_feat['spectral_centroid']
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/audio/test_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.audio.features'`

- [ ] **Step 3: Implement `backend/audio/features.py`**

```python
import numpy as np
import librosa
from backend.audio.loudness import measure_lufs


def extract_features(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Extract audio features needed for rule-based processing decisions.
    audio: (2, N) float32
    Returns dict with float values.
    """
    mono = librosa.to_mono(audio)

    # Dynamic range via RMS frame analysis
    rms_frames = librosa.feature.rms(
        y=mono, frame_length=2048, hop_length=512
    )[0]
    noise_floor_rms = float(np.percentile(rms_frames, 10))
    peak_rms        = float(np.percentile(rms_frames, 95))

    noise_floor_db = 20.0 * np.log10(noise_floor_rms + 1e-9)
    dynamic_range  = 20.0 * np.log10((peak_rms + 1e-9) / (noise_floor_rms + 1e-9))

    # Sibilance: energy ratio 6–10 kHz vs 2–6 kHz
    stft  = np.abs(librosa.stft(mono, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    sib_mask  = (freqs >= 6000) & (freqs <= 10000)
    pres_mask = (freqs >= 2000) & (freqs <=  6000)
    sib_energy  = float(stft[sib_mask].mean())
    pres_energy = float(stft[pres_mask].mean())
    sibilance_ratio = sib_energy / (pres_energy + 1e-9)

    # Spectral centroid (brightness)
    centroid = float(
        librosa.feature.spectral_centroid(y=mono, sr=sample_rate).mean()
    )

    # Zero crossing rate
    zcr = float(librosa.feature.zero_crossing_rate(mono).mean())

    return {
        'lufs':              measure_lufs(audio, sample_rate),
        'noise_floor_db':    float(noise_floor_db),
        'dynamic_range_db':  float(dynamic_range),
        'sibilance_ratio':   float(sibilance_ratio),
        'spectral_centroid': float(centroid),
        'zcr_mean':          float(zcr),
    }
```

- [ ] **Step 4: Run tests and verify they pass**

```powershell
pytest tests/audio/test_features.py -v
```

Expected: All 7 tests `PASSED`

- [ ] **Step 5: Commit**

```powershell
git add backend/audio/features.py tests/audio/test_features.py
git commit -m "feat(audio): add feature extraction module"
```

---

### Task 3: Decision Engine

**Files:**
- Create: `backend/audio/decisions.py`
- Test: `tests/audio/test_decisions.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audio/test_decisions.py`:

```python
import pytest
from backend.audio.decisions import decide_params


def _features(noise_floor_db=-50.0, dynamic_range_db=18.0, sibilance_ratio=1.0,
               spectral_centroid=2500.0, zcr_mean=0.08, lufs=-20.0) -> dict:
    return {
        'noise_floor_db':    noise_floor_db,
        'dynamic_range_db':  dynamic_range_db,
        'sibilance_ratio':   sibilance_ratio,
        'spectral_centroid': spectral_centroid,
        'zcr_mean':          zcr_mean,
        'lufs':              lufs,
    }


def test_decide_params_returns_required_keys():
    params = decide_params(_features())
    required = {
        'gate_threshold_db', 'comp_ratio', 'comp_threshold',
        'comp_attack_ms', 'comp_release_ms', 'presence_freq',
        'presence_gain', 'deess_enabled',
    }
    assert required.issubset(params.keys())


def test_gate_threshold_above_noise_floor():
    features = _features(noise_floor_db=-48.0)
    params = decide_params(features)
    assert params['gate_threshold_db'] > features['noise_floor_db']


def test_gate_threshold_clamped_to_valid_range():
    # Even with very high noise floor, clamp to -30
    params = decide_params(_features(noise_floor_db=-10.0))
    assert params['gate_threshold_db'] <= -30.0
    # Very clean signal, clamp to -70
    params = decide_params(_features(noise_floor_db=-80.0))
    assert params['gate_threshold_db'] >= -70.0


def test_high_dynamic_range_gives_higher_ratio():
    loud_dr  = decide_params(_features(dynamic_range_db=25.0))
    quiet_dr = decide_params(_features(dynamic_range_db=10.0))
    assert loud_dr['comp_ratio'] > quiet_dr['comp_ratio']


def test_deesser_enabled_for_high_sibilance():
    params = decide_params(_features(sibilance_ratio=2.0))
    assert params['deess_enabled'] is True


def test_deesser_disabled_for_low_sibilance():
    params = decide_params(_features(sibilance_ratio=0.8))
    assert params['deess_enabled'] is False


def test_dark_vocal_gets_lower_presence_freq():
    dark   = decide_params(_features(spectral_centroid=1500.0))
    bright = decide_params(_features(spectral_centroid=4000.0))
    assert dark['presence_freq'] < bright['presence_freq']


def test_dark_vocal_gets_higher_presence_gain():
    dark   = decide_params(_features(spectral_centroid=1500.0))
    bright = decide_params(_features(spectral_centroid=4000.0))
    assert dark['presence_gain'] > bright['presence_gain']


def test_high_zcr_reduces_attack_time():
    low_zcr  = decide_params(_features(zcr_mean=0.05))
    high_zcr = decide_params(_features(zcr_mean=0.25))
    assert high_zcr['comp_attack_ms'] < low_zcr['comp_attack_ms']
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/audio/test_decisions.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.audio.decisions'`

- [ ] **Step 3: Implement `backend/audio/decisions.py`**

```python
import numpy as np


def decide_params(features: dict) -> dict:
    """
    Map extracted audio features to DSP processing parameters.
    All rules are explicit and deterministic — no model inference.
    Returns dict of parameter names to values.
    """
    p: dict = {}

    # ── Noise gate ──────────────────────────────────────────────────────────
    p['gate_threshold_db'] = float(
        np.clip(features['noise_floor_db'] + 6.0, -70.0, -30.0)
    )

    # ── Compressor ──────────────────────────────────────────────────────────
    dr = features['dynamic_range_db']
    if dr > 22.0:
        p['comp_ratio']      = 4.0
        p['comp_threshold']  = -20.0
        p['comp_attack_ms']  = 8.0
        p['comp_release_ms'] = 80.0
    elif dr > 15.0:
        p['comp_ratio']      = 3.0
        p['comp_threshold']  = -20.0
        p['comp_attack_ms']  = 12.0
        p['comp_release_ms'] = 120.0
    else:
        p['comp_ratio']      = 2.0
        p['comp_threshold']  = -24.0
        p['comp_attack_ms']  = 18.0
        p['comp_release_ms'] = 160.0

    # High transient density → faster attack
    if features['zcr_mean'] > 0.15:
        p['comp_attack_ms'] = max(4.0, p['comp_attack_ms'] - 6.0)

    # ── Presence EQ ─────────────────────────────────────────────────────────
    centroid = features['spectral_centroid']
    if centroid < 2000.0:
        p['presence_freq'] = 3500.0
        p['presence_gain'] = 3.0
    elif centroid < 3000.0:
        p['presence_freq'] = 4000.0
        p['presence_gain'] = 2.0
    else:
        p['presence_freq'] = 5000.0
        p['presence_gain'] = 1.5

    # ── De-esser ────────────────────────────────────────────────────────────
    sr = features['sibilance_ratio']
    if sr > 1.5:
        p['deess_enabled']   = True
        p['deess_freq']      = 7000.0
        p['deess_threshold'] = -16.0
        p['deess_reduction'] = -6.0
    elif sr > 1.2:
        p['deess_enabled']   = True
        p['deess_freq']      = 8000.0
        p['deess_threshold'] = -22.0
        p['deess_reduction'] = -4.0
    else:
        p['deess_enabled'] = False

    return p
```

- [ ] **Step 4: Run tests and verify they pass**

```powershell
pytest tests/audio/test_decisions.py -v
```

Expected: All 9 tests `PASSED`

- [ ] **Step 5: Commit**

```powershell
git add backend/audio/decisions.py tests/audio/test_decisions.py
git commit -m "feat(audio): add rule-based parameter decision engine"
```

---

### Task 4: De-esser Module

**Files:**
- Create: `backend/audio/deesser.py`
- Test: `tests/audio/test_deesser.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audio/test_deesser.py`:

```python
import numpy as np
import pytest
from backend.audio.deesser import apply_deesser

SAMPLE_RATE = 44100


def _sibilant_audio(duration_s: float = 2.0) -> np.ndarray:
    """High-frequency heavy audio (simulates harsh sibilance)."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (0.4 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32)
    mono += (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.stack([mono, mono])


def _low_freq_audio(duration_s: float = 2.0) -> np.ndarray:
    """Low-frequency audio (no sibilance)."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    return np.stack([mono, mono])


def test_deesser_returns_same_shape():
    audio = _sibilant_audio()
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    assert result.shape == audio.shape


def test_deesser_returns_float32():
    audio = _sibilant_audio()
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    assert result.dtype == np.float32


def test_deesser_reduces_high_frequency_energy():
    audio = _sibilant_audio()
    import librosa
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    sib_mask = freqs >= 7000

    before_stft = np.abs(librosa.stft(librosa.to_mono(audio), n_fft=2048))
    after_stft  = np.abs(librosa.stft(librosa.to_mono(result), n_fft=2048))

    before_energy = float(before_stft[sib_mask].mean())
    after_energy  = float(after_stft[sib_mask].mean())

    assert after_energy < before_energy, (
        f"De-esser did not reduce high-freq energy: before={before_energy:.4f}, "
        f"after={after_energy:.4f}"
    )


def test_deesser_does_not_touch_low_freq_audio_significantly():
    audio = _low_freq_audio()
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    # Low-freq audio should pass through with < 0.5 dB change
    before_rms = float(np.sqrt(np.mean(audio ** 2)))
    after_rms  = float(np.sqrt(np.mean(result ** 2)))
    diff_db = abs(20.0 * np.log10((after_rms + 1e-9) / (before_rms + 1e-9)))
    assert diff_db < 1.0, f"De-esser changed low-freq audio by {diff_db:.2f} dB"


def test_deesser_output_has_no_clipping():
    audio = _sibilant_audio()
    result = apply_deesser(audio, SAMPLE_RATE, freq=7000.0,
                           threshold_db=-16.0, reduction_db=-6.0)
    assert float(np.abs(result).max()) <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/audio/test_deesser.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.audio.deesser'`

- [ ] **Step 3: Implement `backend/audio/deesser.py`**

```python
import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d


def apply_deesser(audio: np.ndarray, sample_rate: int, freq: float,
                  threshold_db: float, reduction_db: float) -> np.ndarray:
    """
    Spectral de-esser: detect frames where energy above `freq` Hz exceeds
    `threshold_db`, apply `reduction_db` gain to those frames on all channels.

    audio:          (2, N) float32
    freq:           frequency above which sibilance is measured (e.g. 7000.0)
    threshold_db:   level above which de-essing activates (e.g. -16.0)
    reduction_db:   gain reduction applied (negative, e.g. -6.0)
    Returns:        (2, N) float32, same shape as input
    """
    n_fft = 2048
    hop   = 512

    mono  = librosa.to_mono(audio)
    stft  = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    sib_mask    = freqs >= freq
    magnitudes  = np.abs(stft)
    sib_energy  = magnitudes[sib_mask].mean(axis=0)       # (n_frames,)

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    reduction_linear = 10.0 ** (reduction_db / 20.0)      # < 1.0

    # Gain = 1.0 below threshold, reduction_linear above
    gain_curve = np.where(sib_energy > threshold_linear, reduction_linear, 1.0)

    # Smooth to prevent zipper noise
    gain_curve = uniform_filter1d(gain_curve.astype(np.float64), size=9).astype(np.float32)

    # Map frame-level gain to sample-level gain via linear interpolation
    frame_centers = np.arange(len(gain_curve)) * hop + n_fft // 2
    sample_indices = np.arange(audio.shape[1], dtype=np.float32)
    gain_samples = np.interp(sample_indices, frame_centers, gain_curve)
    gain_samples = np.clip(gain_samples, float(reduction_linear), 1.0).astype(np.float32)

    return (audio * gain_samples[np.newaxis, :]).astype(np.float32)
```

- [ ] **Step 4: Run tests and verify they pass**

```powershell
pytest tests/audio/test_deesser.py -v
```

Expected: All 5 tests `PASSED`

- [ ] **Step 5: Commit**

```powershell
git add backend/audio/deesser.py tests/audio/test_deesser.py
git commit -m "feat(audio): add spectral de-esser module"
```

---

### Task 5: Vocal Processing Chain

**Files:**
- Create: `backend/audio/vocal_chain.py`
- Test: `tests/audio/test_vocal_chain.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audio/test_vocal_chain.py`:

```python
import numpy as np
import pytest
import librosa
from backend.audio.vocal_chain import apply_vocal_chain

SAMPLE_RATE = 44100


def _noisy_vocal(duration_s: float = 3.0) -> np.ndarray:
    """Simulate a noisy vocal: quiet noise floor + 440 Hz signal + harsh highs."""
    rng = np.random.default_rng(99)
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    noise  = 0.02 * rng.standard_normal(len(t)).astype(np.float32)
    voice  = 0.35 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sib    = 0.15 * np.sin(2 * np.pi * 8000 * t).astype(np.float32)
    mono   = noise + voice + sib
    return np.stack([mono, mono])


_DEFAULT_PARAMS = {
    'gate_threshold_db':  -46.0,
    'comp_ratio':         3.0,
    'comp_threshold':     -20.0,
    'comp_attack_ms':     12.0,
    'comp_release_ms':    120.0,
    'presence_freq':      4000.0,
    'presence_gain':      2.0,
    'deess_enabled':      True,
    'deess_freq':         7000.0,
    'deess_threshold':    -16.0,
    'deess_reduction':    -6.0,
}


def test_vocal_chain_returns_same_shape():
    audio = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, _DEFAULT_PARAMS)
    assert result.shape == audio.shape


def test_vocal_chain_returns_float32():
    audio = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, _DEFAULT_PARAMS)
    assert result.dtype == np.float32


def test_vocal_chain_no_clipping():
    audio = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, _DEFAULT_PARAMS)
    assert float(np.abs(result).max()) <= 1.05, "Output clips significantly"


def test_vocal_chain_reduces_noise_floor():
    audio = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, _DEFAULT_PARAMS)
    mono_in  = librosa.to_mono(audio)
    mono_out = librosa.to_mono(result)
    rms_in  = np.percentile(librosa.feature.rms(y=mono_in)[0],  10)
    rms_out = np.percentile(librosa.feature.rms(y=mono_out)[0], 10)
    assert rms_out <= rms_in * 1.1, "Noise floor did not decrease"


def test_vocal_chain_without_deesser():
    params = {**_DEFAULT_PARAMS, 'deess_enabled': False}
    audio  = _noisy_vocal()
    result = apply_vocal_chain(audio, SAMPLE_RATE, params)
    assert result.shape == audio.shape
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/audio/test_vocal_chain.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.audio.vocal_chain'`

- [ ] **Step 3: Implement `backend/audio/vocal_chain.py`**

```python
import numpy as np
import noisereduce as nr
from pedalboard import (
    Pedalboard, HighpassFilter, LowShelfFilter, HighShelfFilter,
    Compressor, NoiseGate,
)
from backend.audio.deesser import apply_deesser


def apply_vocal_chain(audio: np.ndarray, sample_rate: int,
                      params: dict) -> np.ndarray:
    """
    Full vocal processing chain. Order matters:
    1. Noise reduction
    2. Highpass filter (80 Hz)
    3. Noise gate
    4. Low-mid cut (300 Hz mud reduction)
    5. Presence boost (freq/gain from params)
    6. Air shelf (+1.5 dB at 12 kHz)
    7. Compressor
    8. De-esser (if enabled)

    audio:   (2, N) float32
    params:  dict from decide_params()
    Returns: (2, N) float32
    """
    # 1. Noise reduction — use first 300 ms as noise profile
    noise_len = min(int(sample_rate * 0.3), audio.shape[1] // 4)
    noise_sample = audio[:, :noise_len]
    audio = nr.reduce_noise(
        y=audio, sr=sample_rate, y_noise=noise_sample,
        prop_decrease=0.75, n_fft=2048,
    ).astype(np.float32)

    # 2–7. EQ, gate, compressor via pedalboard
    chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80.0),
        NoiseGate(
            threshold_db=params['gate_threshold_db'],
            ratio=10.0,
            attack_ms=1.0,
            release_ms=80.0,
        ),
        LowShelfFilter(
            cutoff_frequency_hz=300.0,
            gain_db=-2.0,
        ),
        HighShelfFilter(
            cutoff_frequency_hz=params['presence_freq'],
            gain_db=params['presence_gain'],
        ),
        HighShelfFilter(
            cutoff_frequency_hz=12000.0,
            gain_db=1.5,
        ),
        Compressor(
            threshold_db=params['comp_threshold'],
            ratio=params['comp_ratio'],
            attack_ms=params['comp_attack_ms'],
            release_ms=params['comp_release_ms'],
        ),
    ])
    audio = chain(audio, sample_rate).astype(np.float32)

    # 8. De-esser
    if params.get('deess_enabled'):
        audio = apply_deesser(
            audio, sample_rate,
            freq=params['deess_freq'],
            threshold_db=params['deess_threshold'],
            reduction_db=params['deess_reduction'],
        )

    return audio
```

- [ ] **Step 4: Run tests and verify they pass**

```powershell
pytest tests/audio/test_vocal_chain.py -v
```

Expected: All 5 tests `PASSED`

- [ ] **Step 5: Commit**

```powershell
git add backend/audio/vocal_chain.py tests/audio/test_vocal_chain.py
git commit -m "feat(audio): add full vocal processing chain"
```

---

### Task 6: Mixer and Master Chain

**Files:**
- Create: `backend/audio/mixer.py`
- Create: `backend/audio/master.py`
- Test: `tests/audio/test_mixer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audio/test_mixer.py`:

```python
import numpy as np
import pytest
from backend.audio.loudness import measure_lufs
from backend.audio.mixer import balance_and_sum
from backend.audio.master import apply_master_chain

SAMPLE_RATE = 44100


def _stereo_sine(freq: float = 440.0, duration_s: float = 4.0,
                 amplitude: float = 0.4) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono])


def test_balance_and_sum_returns_stereo():
    beat  = _stereo_sine(freq=80.0)
    vocal = _stereo_sine(freq=440.0)
    result = balance_and_sum(beat, vocal, SAMPLE_RATE)
    assert result.ndim == 2
    assert result.shape[0] == 2


def test_balance_and_sum_handles_length_mismatch():
    beat  = _stereo_sine(duration_s=4.0)
    vocal = _stereo_sine(duration_s=2.5)  # shorter vocal
    result = balance_and_sum(beat, vocal, SAMPLE_RATE)
    # Output length should equal longer input
    expected_len = beat.shape[1]
    assert result.shape[1] == expected_len


def test_balance_and_sum_beat_louder_than_vocal():
    beat  = _stereo_sine(freq=80.0,  amplitude=0.5)
    vocal = _stereo_sine(freq=440.0, amplitude=0.1)
    beat_norm, vocal_norm, _ = balance_and_sum(
        beat, vocal, SAMPLE_RATE, return_stems=True
    )
    beat_lufs  = measure_lufs(beat_norm,  SAMPLE_RATE)
    vocal_lufs = measure_lufs(vocal_norm, SAMPLE_RATE)
    # Beat should be louder by roughly 4 dB
    assert beat_lufs > vocal_lufs, "Beat should be louder than vocal in mix"


def test_master_chain_reaches_target_lufs():
    mix = _stereo_sine(amplitude=0.3, duration_s=5.0)
    mastered = apply_master_chain(mix, SAMPLE_RATE, target_lufs=-14.0)
    result_lufs = measure_lufs(mastered, SAMPLE_RATE)
    assert abs(result_lufs - (-14.0)) < 1.5, f"Expected ~-14 LUFS, got {result_lufs:.2f}"


def test_master_chain_true_peak_under_limit():
    mix = _stereo_sine(amplitude=0.9, duration_s=4.0)
    mastered = apply_master_chain(mix, SAMPLE_RATE, target_lufs=-14.0)
    # Allow up to -0.5 dBFS true peak (limiter accuracy tolerance)
    true_peak_dbfs = 20.0 * np.log10(float(np.abs(mastered).max()) + 1e-9)
    assert true_peak_dbfs <= -0.5, f"True peak too high: {true_peak_dbfs:.2f} dBFS"


def test_master_chain_returns_float32():
    mix = _stereo_sine()
    result = apply_master_chain(mix, SAMPLE_RATE, target_lufs=-14.0)
    assert result.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/audio/test_mixer.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.audio.mixer'`

- [ ] **Step 3: Implement `backend/audio/mixer.py`**

```python
import numpy as np
from backend.audio.loudness import normalize_to_lufs

TARGET_BEAT_LUFS  = -16.0
TARGET_VOCAL_LUFS = -20.0


def _pad_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[1] >= length:
        return audio
    pad_width = length - audio.shape[1]
    return np.pad(audio, ((0, 0), (0, pad_width)))


def balance_and_sum(beat: np.ndarray, vocal: np.ndarray,
                    sample_rate: int,
                    beat_target_lufs: float = TARGET_BEAT_LUFS,
                    vocal_target_lufs: float = TARGET_VOCAL_LUFS,
                    return_stems: bool = False):
    """
    Normalize beat and vocal to target LUFS values, pad to same length, sum.

    Returns:
      - mix (2, N) if return_stems=False
      - (beat_norm, vocal_norm, mix) if return_stems=True
    """
    beat_norm  = normalize_to_lufs(beat,  sample_rate, beat_target_lufs)
    vocal_norm = normalize_to_lufs(vocal, sample_rate, vocal_target_lufs)

    max_len    = max(beat_norm.shape[1], vocal_norm.shape[1])
    beat_pad   = _pad_to_length(beat_norm,  max_len)
    vocal_pad  = _pad_to_length(vocal_norm, max_len)
    mix        = (beat_pad + vocal_pad).astype(np.float32)

    if return_stems:
        return beat_norm, vocal_norm, mix
    return mix
```

- [ ] **Step 4: Implement `backend/audio/master.py`**

```python
import numpy as np
from pedalboard import Pedalboard, Limiter
from backend.audio.loudness import normalize_to_lufs

TRUE_PEAK_LIMIT_DB = -1.0


def apply_master_chain(mix: np.ndarray, sample_rate: int,
                       target_lufs: float = -14.0) -> np.ndarray:
    """
    Mastering chain: limit true peak to -1 dBTP, then normalize to target_lufs.
    Two limiter passes ensure no overshoot after normalization.

    mix:  (2, N) float32
    Returns: (2, N) float32
    """
    limiter = Pedalboard([
        Limiter(threshold_db=TRUE_PEAK_LIMIT_DB, release_ms=100.0)
    ])

    mix = limiter(mix, sample_rate).astype(np.float32)
    mix = normalize_to_lufs(mix, sample_rate, target_lufs)
    mix = limiter(mix, sample_rate).astype(np.float32)

    return mix
```

- [ ] **Step 5: Run tests and verify they pass**

```powershell
pytest tests/audio/test_mixer.py -v
```

Expected: All 6 tests `PASSED`

- [ ] **Step 6: Commit**

```powershell
git add backend/audio/mixer.py backend/audio/master.py tests/audio/test_mixer.py
git commit -m "feat(audio): add mixer gain staging and master chain"
```

---

### Task 7: Audio I/O Helpers

**Files:**
- Create: `backend/audio/io.py`
- Test: `tests/audio/test_io.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audio/test_io.py`:

```python
import numpy as np
import pytest
import tempfile
import os
from backend.audio.io import load_audio, save_audio

SAMPLE_RATE = 44100


def _make_temp_wav(duration_s: float = 1.0) -> str:
    """Write a temp 440 Hz sine WAV at 44100 Hz stereo, return path."""
    import soundfile as sf
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([mono, mono]).T  # soundfile wants (N, 2)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(tmp.name, stereo, SAMPLE_RATE, subtype='PCM_16')
    return tmp.name


def test_load_audio_returns_stereo_float32():
    path = _make_temp_wav()
    try:
        audio = load_audio(path)
        assert audio.ndim == 2
        assert audio.shape[0] == 2
        assert audio.dtype == np.float32
    finally:
        os.unlink(path)


def test_load_audio_resamples_to_44100():
    import soundfile as sf
    # Write a 22050 Hz file
    t = np.linspace(0, 1.0, 22050, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(tmp.name, mono, 22050)
    try:
        audio = load_audio(tmp.name)
        assert audio.shape[1] == SAMPLE_RATE  # resampled to 44100 samples for 1s
    finally:
        os.unlink(tmp.name)


def test_save_audio_creates_file():
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio = np.stack([mono, mono])
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    os.unlink(tmp.name)
    try:
        save_audio(tmp.name, audio, SAMPLE_RATE)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 1000
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def test_load_mono_converts_to_stereo():
    import soundfile as sf
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(tmp.name, mono, SAMPLE_RATE)
    try:
        audio = load_audio(tmp.name)
        assert audio.shape[0] == 2, "Mono input should be converted to stereo"
    finally:
        os.unlink(tmp.name)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/audio/test_io.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.audio.io'`

- [ ] **Step 3: Implement `backend/audio/io.py`**

```python
import numpy as np
import librosa
import soundfile as sf

TARGET_SAMPLE_RATE = 44100


def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file, resample to target_sr, convert to (2, N) float32.
    Handles mono input by duplicating to stereo.
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    return audio.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Save (2, N) float32 as 24-bit stereo WAV."""
    sf.write(path, audio.T, sample_rate, subtype='PCM_24')
```

- [ ] **Step 4: Run tests and verify they pass**

```powershell
pytest tests/audio/test_io.py -v
```

Expected: All 4 tests `PASSED`

- [ ] **Step 5: Commit**

```powershell
git add backend/audio/io.py tests/audio/test_io.py
git commit -m "feat(audio): add audio I/O helpers with auto-resampling"
```

---

### Task 8: Prototype CLI Script

**Files:**
- Create: `scripts/prototype.py`
- Test: integration test via CLI (manual + one automated test)

- [ ] **Step 1: Run the full test suite first to confirm all modules pass**

```powershell
pytest tests/audio/ -v --tb=short
```

Expected: All tests `PASSED`

- [ ] **Step 2: Write `scripts/prototype.py`**

```python
#!/usr/bin/env python3
"""
BeatFlow Stage 1 Prototype
Usage: python scripts/prototype.py beat.wav vocal.wav output.wav
"""
import sys
from pathlib import Path

# Add project root to path so backend imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from backend.audio.io        import load_audio, save_audio, TARGET_SAMPLE_RATE
from backend.audio.loudness  import measure_lufs
from backend.audio.features  import extract_features
from backend.audio.decisions import decide_params
from backend.audio.vocal_chain import apply_vocal_chain
from backend.audio.mixer     import balance_and_sum
from backend.audio.master    import apply_master_chain


def _print_section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def run(beat_path: str, vocal_path: str, output_path: str) -> None:
    SR = TARGET_SAMPLE_RATE

    _print_section("Loading audio")
    beat  = load_audio(beat_path)
    vocal = load_audio(vocal_path)
    print(f"  Beat  : {beat.shape[1] / SR:.1f}s  "
          f"LUFS={measure_lufs(beat, SR):.1f}")
    print(f"  Vocal : {vocal.shape[1] / SR:.1f}s  "
          f"LUFS={measure_lufs(vocal, SR):.1f}")

    _print_section("Analyzing vocal")
    features = extract_features(vocal, SR)
    print(f"  LUFS             : {features['lufs']:.1f}")
    print(f"  Noise floor      : {features['noise_floor_db']:.1f} dB")
    print(f"  Dynamic range    : {features['dynamic_range_db']:.1f} dB")
    print(f"  Sibilance ratio  : {features['sibilance_ratio']:.2f}")
    print(f"  Spectral centroid: {features['spectral_centroid']:.0f} Hz")
    print(f"  ZCR mean         : {features['zcr_mean']:.4f}")

    _print_section("Deciding parameters")
    params = decide_params(features)
    print(f"  Gate threshold   : {params['gate_threshold_db']:.1f} dB")
    print(f"  Comp ratio       : {params['comp_ratio']:.1f}:1")
    print(f"  Comp threshold   : {params['comp_threshold']:.1f} dB")
    print(f"  Comp attack      : {params['comp_attack_ms']:.0f} ms")
    print(f"  Presence boost   : +{params['presence_gain']:.1f} dB @ {params['presence_freq']:.0f} Hz")
    print(f"  De-esser         : {'ON' if params['deess_enabled'] else 'OFF'}")
    if params.get('deess_enabled'):
        print(f"    Freq           : {params['deess_freq']:.0f} Hz")
        print(f"    Threshold      : {params['deess_threshold']:.1f} dB")
        print(f"    Reduction      : {params['deess_reduction']:.1f} dB")

    _print_section("Processing vocal")
    vocal_processed = apply_vocal_chain(vocal, SR, params)

    _print_section("Balancing and summing")
    mix = balance_and_sum(beat, vocal_processed, SR)

    _print_section("Mastering")
    master = apply_master_chain(mix, SR, target_lufs=-14.0)

    _print_section("Exporting")
    save_audio(output_path, master, SR)

    # ── Before/after report ───────────────────────────────────────────
    after_features = extract_features(vocal_processed, SR)
    _print_section("Before / After Report")
    print(f"  {'Metric':<25}  {'Before':>10}  {'After':>10}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}")
    print(f"  {'Vocal LUFS':<25}  {features['lufs']:>10.1f}  "
          f"{measure_lufs(vocal_processed, SR):>10.1f}")
    print(f"  {'Dynamic range (dB)':<25}  {features['dynamic_range_db']:>10.1f}  "
          f"{after_features['dynamic_range_db']:>10.1f}")
    print(f"  {'Sibilance ratio':<25}  {features['sibilance_ratio']:>10.2f}  "
          f"{after_features['sibilance_ratio']:>10.2f}")
    print(f"\n  Final mix LUFS   : {measure_lufs(master, SR):.1f}")
    print(f"  True peak (dBFS) : {20 * np.log10(float(np.abs(master).max()) + 1e-9):.2f}")
    print(f"  Saved to         : {output_path}")
    print()


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python scripts/prototype.py beat.wav vocal.wav output.wav")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write an integration test using synthetic WAVs**

Create `tests/audio/test_prototype_integration.py`:

```python
import numpy as np
import os
import tempfile
import soundfile as sf
import pytest
from scripts.prototype import run

SAMPLE_RATE = 44100


def _write_wav(path: str, freq: float = 440.0, duration_s: float = 4.0,
               amplitude: float = 0.35) -> None:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    # Add a noise floor
    rng = np.random.default_rng(42)
    mono += 0.01 * rng.standard_normal(len(mono)).astype(np.float32)
    stereo = np.stack([mono, mono]).T
    sf.write(path, stereo, SAMPLE_RATE, subtype='PCM_16')


def test_prototype_run_produces_output_file():
    with tempfile.TemporaryDirectory() as tmp:
        beat_path   = os.path.join(tmp, 'beat.wav')
        vocal_path  = os.path.join(tmp, 'vocal.wav')
        output_path = os.path.join(tmp, 'output.wav')

        _write_wav(beat_path,  freq=80.0,  amplitude=0.5)
        _write_wav(vocal_path, freq=440.0, amplitude=0.3)

        run(beat_path, vocal_path, output_path)

        assert os.path.exists(output_path), "Output file was not created"
        assert os.path.getsize(output_path) > 10_000, "Output file is suspiciously small"


def test_prototype_output_lufs_near_target():
    import pyloudnorm as pyln
    with tempfile.TemporaryDirectory() as tmp:
        beat_path   = os.path.join(tmp, 'beat.wav')
        vocal_path  = os.path.join(tmp, 'vocal.wav')
        output_path = os.path.join(tmp, 'output.wav')

        _write_wav(beat_path,  freq=80.0,  amplitude=0.5, duration_s=5.0)
        _write_wav(vocal_path, freq=440.0, amplitude=0.3, duration_s=5.0)

        run(beat_path, vocal_path, output_path)

        audio, sr = sf.read(output_path)
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(audio)
        assert abs(lufs - (-14.0)) < 2.0, f"Final LUFS {lufs:.1f} not near -14"


def test_prototype_output_no_clipping():
    with tempfile.TemporaryDirectory() as tmp:
        beat_path   = os.path.join(tmp, 'beat.wav')
        vocal_path  = os.path.join(tmp, 'vocal.wav')
        output_path = os.path.join(tmp, 'output.wav')

        _write_wav(beat_path,  freq=80.0,  amplitude=0.8, duration_s=5.0)
        _write_wav(vocal_path, freq=440.0, amplitude=0.8, duration_s=5.0)

        run(beat_path, vocal_path, output_path)

        audio, _ = sf.read(output_path)
        max_sample = float(np.abs(audio).max())
        assert max_sample <= 1.0, f"Output clips at {max_sample:.4f}"
```

- [ ] **Step 4: Run the integration tests**

```powershell
pytest tests/audio/test_prototype_integration.py -v
```

Expected: All 3 tests `PASSED` (this takes ~10–20s due to audio processing)

- [ ] **Step 5: Run the full test suite**

```powershell
pytest tests/ -v --tb=short
```

Expected: All tests `PASSED`

- [ ] **Step 6: Smoke test with real audio (manual)**

```powershell
# Replace with your actual test files
python scripts/prototype.py tests/fixtures/beats/test_beat.wav tests/fixtures/vocals/bad/noisy_vocal.wav output_test.wav
```

Listen to `output_test.wav` and compare to the raw mix. The vocal should sound cleaner and the levels should be balanced.

- [ ] **Step 7: Commit**

```powershell
git add scripts/prototype.py tests/audio/test_prototype_integration.py
git commit -m "feat: add Stage 1 prototype CLI — vocal chain + mix balance + master"
```

---

### Task 9: Configs and .gitignore

**Files:**
- Create: `configs/processing_presets.yaml`
- Create: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
*.pyo
.env
uploads/
outputs/
*.wav
*.mp3
*.flac
.pytest_cache/
.coverage
htmlcov/
dist/
*.egg-info/
```

- [ ] **Step 2: Create `configs/processing_presets.yaml`**

```yaml
# Named presets for genre-aware processing (Stage 6)
# Values override the decision engine defaults for the specified genre

hip_hop:
  beat_target_lufs: -15.0
  vocal_target_lufs: -19.0
  mix_target_lufs: -14.0
  comp_ratio_override: 4.0
  presence_freq: 3500.0
  presence_gain: 3.0
  reverb_room_size: 0.3
  reverb_wet: 0.15

rnb:
  beat_target_lufs: -16.0
  vocal_target_lufs: -20.0
  mix_target_lufs: -14.0
  comp_ratio_override: null     # use decision engine
  presence_freq: 4500.0
  presence_gain: 2.0
  reverb_room_size: 0.5
  reverb_wet: 0.25

pop:
  beat_target_lufs: -16.0
  vocal_target_lufs: -20.0
  mix_target_lufs: -13.0
  comp_ratio_override: null
  presence_freq: 5000.0
  presence_gain: 1.5
  reverb_room_size: 0.4
  reverb_wet: 0.2
```

- [ ] **Step 3: Create `.env.example`**

```
# BeatFlow environment variables
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
DATABASE_URL=sqlite:///./beatflow.db
REDIS_URL=redis://localhost:6379/0
MAX_UPLOAD_SIZE_MB=100
TARGET_MIX_LUFS=-14.0
TARGET_BEAT_LUFS=-16.0
TARGET_VOCAL_LUFS=-20.0

# Stage 6+: LLM parameter suggestions (optional)
# ANTHROPIC_API_KEY=your_key_here
```

- [ ] **Step 4: Commit**

```powershell
git add configs/ .gitignore .env.example
git commit -m "chore: add processing presets, gitignore, and env example"
```

---

## Stage 1 Complete — Verification Checklist

Run this checklist after completing all tasks:

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `python scripts/prototype.py beat.wav vocal.wav out.wav` — runs without error
- [ ] Output LUFS is within ±1.5 dB of -14.0
- [ ] Output has no samples ≥ 1.0 (check: `python -c "import soundfile as sf, numpy as np; a,_ = sf.read('out.wav'); print(np.abs(a).max())"`)
- [ ] Console output shows before/after report
- [ ] Listen to output: vocal is cleaner, better balanced against beat

---

## Next Steps After Stage 1

1. **Collect real test audio**: Add 5 good vocals + 5 bad vocals + 3 beats to `tests/fixtures/`
2. **Manual rating session**: Process each vocal/beat pair, listen, rate using the 3-axis rubric
3. **Tune decision thresholds**: Adjust `decide_params()` based on what sounds good/bad on real material
4. **Stage 3 begins**: Wrap prototype in FastAPI (`backend/main.py`) with a `/process` endpoint
