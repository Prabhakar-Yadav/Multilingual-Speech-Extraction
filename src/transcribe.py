"""Phase A - ASR: speech -> text in the spoken language.

faster-whisper (CTranslate2, CPU int8, no PyTorch).
Audio is expected to be Bengali, Hindi, or English. Language is auto-detected
but constrained to {bn, hi, en}: if Whisper misdetects to some other language,
we re-run forcing the most probable of the three. This keeps the same code
working for both Bengali and Hindi without hard-coding either.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import time
from pathlib import Path

from faster_whisper import WhisperModel

ALLOWED = {"bn", "hi", "en"}  # the only languages we expect


def _best_allowed(all_language_probs):
    """Pick the most probable language within ALLOWED from the full distribution."""
    best = None
    for lang, prob in (all_language_probs or []):
        if lang in ALLOWED and (best is None or prob > best[1]):
            best = (lang, prob)
    return best


def _run(model, audio_path, language, beam_size):
    return model.transcribe(
        audio_path,
        language=language,        # None => auto-detect
        task="transcribe",        # keep source language; translation is Phase B
        beam_size=beam_size,      # 1 (greedy) = lowest memory; higher = better but more RAM
        vad_filter=True,          # Silero VAD: drop silence/noise
        vad_parameters=dict(min_silence_duration_ms=500),
        # Anti-hallucination: on low-quality audio Whisper invents text. These make
        # it stay quiet on unintelligible/silent parts instead of looping garbage.
        condition_on_previous_text=False,   # key: stops "so so so" repetition loops
        compression_ratio_threshold=2.4,    # drop segments that look like gibberish
        log_prob_threshold=-1.0,            # drop low-confidence segments
        no_speech_threshold=0.6,
    )


def transcribe(audio_path: str, model_size: str = "medium", language: str | None = None,
               beam_size: int = 1, cpu_threads: int = 4):
    # int8 + few threads keeps RAM low enough for the 8 GB target machine; CPU only.
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)

    t0 = time.time()
    segments, info = _run(model, audio_path, language, beam_size)

    # Constrain auto-detection to bn/hi/en. (info is ready before we consume
    # segments; all_language_probs is only populated when language is None.)
    if language is None and info.language not in ALLOWED:
        forced = _best_allowed(getattr(info, "all_language_probs", None))
        if forced:
            print(f"  (auto-detected '{info.language}' not in bn/hi/en -> forcing '{forced[0]}')")
            segments, info = _run(model, audio_path, forced[0], beam_size)

    print(f"Detected language : {info.language} (p={info.language_probability:.3f})")
    print(f"Audio duration    : {info.duration:.1f}s\n--- live transcript ---")

    parts = []
    for seg in segments:                      # consume the generator (does the work)
        line = seg.text.strip()
        print(f"[{seg.start:6.1f}s] {line}")
        parts.append(line)

    elapsed = time.time() - t0
    return {
        "audio": audio_path,
        "model": f"faster-whisper-{model_size}",
        "detected_language": info.language,
        "language_probability": round(info.language_probability, 3),
        "audio_duration_sec": round(info.duration, 1),
        "processing_sec": round(elapsed, 1),
        "transcript": " ".join(parts).strip(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="medium", help="tiny|base|small|medium or local path")
    ap.add_argument("--language", default=None, help="force bn|hi|en, else auto-detect")
    ap.add_argument("--beam_size", type=int, default=1, help="1=greedy/lowest RAM; 5=better/more RAM")
    ap.add_argument("--cpu_threads", type=int, default=4)
    ap.add_argument("--out", default="outputs/transcript.json")
    args = ap.parse_args()

    result = transcribe(args.audio, args.model, args.language, args.beam_size, args.cpu_threads)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nProcessing time   : {result['processing_sec']}s")
    print(f"Saved transcript  : {args.out}")
