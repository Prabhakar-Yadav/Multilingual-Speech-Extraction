"""
English medical ASR - on-prem, CPU-only.

Uses faster-whisper (CTranslate2 + int8), which is ~2x faster than vanilla Whisper
on CPU and needs far less RAM. Nothing leaves the machine.

    python transcribe.py audio.wav
    python transcribe.py audio.wav --model small.en --device cpu

Model choices (English-only .en models are more accurate AND smaller for English):
    tiny.en    ~ 75 MB   fastest, lowest accuracy
    base.en    ~145 MB
    small.en   ~ 480 MB  <- good speed/accuracy balance on CPU
    medium.en  ~1.5 GB
    distil-large-v3 ~1.5 GB  <- best accuracy, still CPU-friendly (recommended)
"""
import argparse, time
from pathlib import Path


def load_model(name: str, device: str = "cpu", compute_type: str | None = None):
    from faster_whisper import WhisperModel
    # int8 on CPU = big speedup, minimal accuracy loss
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "float16"
    print(f"loading {name} ({device}, {compute_type})...", flush=True)
    return WhisperModel(name, device=device, compute_type=compute_type)


def transcribe(model, audio_path: str, language: str = "en"):
    """Returns (full_text, segments, seconds_taken, audio_duration)."""
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,                 # skip silence -> faster + fewer hallucinations
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,  # stops repetition loops on long audio
    )
    segs = list(segments)                 # generator -> list (this is where work happens)
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, segs, time.time() - t0, info.duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="distil-large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write transcript to this file")
    a = ap.parse_args()

    if not Path(a.audio).exists():
        raise SystemExit(f"no such file: {a.audio}")

    model = load_model(a.model, a.device)
    text, segs, took, dur = transcribe(model, a.audio)

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)
    print(f"audio {dur:.1f}s | transcribed in {took:.1f}s | "
          f"{dur/took:.1f}x real-time | {len(segs)} segments")

    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
