"""Transcribe with a HuggingFace (transformers) Whisper model, e.g. a Bengali-
fine-tuned one. Used to test Bengali-specialized ASR on the cleaned audio.

    python src/transcribe_hf.py <model_dir> <audio.wav> <out.json>
"""
import sys
import json
import time
import wave
from pathlib import Path

import numpy as np
from transformers import pipeline

model_dir = sys.argv[1]
audio = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "outputs/transcript_hf.json"

# Load the WAV (we feed raw samples so no ffmpeg/soundfile dependency is needed).
with wave.open(audio, "rb") as w:
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

asr = pipeline(
    "automatic-speech-recognition",
    model=model_dir,
    chunk_length_s=30,      # chunk long audio
    stride_length_s=5,
    device="cpu",
)

t0 = time.time()
res = asr(
    {"raw": x, "sampling_rate": sr},
    # Plain decode worked best on this very degraded audio. (no_repeat_ngram /
    # repetition_penalty made it WORSE here - forcing non-repetition on
    # unintelligible audio just varies the garbage. Revisit for clean audio.)
    generate_kwargs={"language": "bn", "task": "transcribe"},
    batch_size=1,
)
elapsed = round(time.time() - t0, 1)

text = res["text"].strip()
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(
    {"model": model_dir, "audio": audio, "processing_sec": elapsed, "transcript": text},
    ensure_ascii=False, indent=2), encoding="utf-8")

print(f"model: {model_dir}")
print(f"processing: {elapsed}s")
print("--- TRANSCRIPT ---")
print(text)
