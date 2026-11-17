"""
Benchmark English ASR models on PriMock57 (real doctor-patient consultations).

Reports WER + CER + speed for each model, measured against the dataset's HUMAN
transcripts - so the numbers are honest and comparable.

    python benchmark_asr.py                 # whisper models, CPU
    python benchmark_asr.py --n 10          # only 10 consultations (quick)
    python benchmark_asr.py --with-parakeet # also test Parakeet TDT (needs nemo)
    python benchmark_asr.py --device cuda   # when the GPU is free

Lessons applied from the Bengali work:
  * WER is an ERROR rate - LOWER IS BETTER. We print accuracy too, to avoid confusion.
  * Report CER as well: it's the fairer metric when a single wrong letter kills a word.
  * Never score on data a model was tuned on - PriMock57 is pure held-out test data.
"""
import argparse, json, re, time
from pathlib import Path

import jiwer

HERE = Path(__file__).resolve().parent
REPO = HERE / "data" / "primock57"

# WER is very sensitive to formatting; normalise both sides the same way so we
# measure real recognition errors, not punctuation/casing differences.
NORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def load_pairs(limit=None):
    """Find (audio, reference_text) pairs in the PriMock57 repo."""
    pairs = []
    for wav in sorted(REPO.rglob("*.wav")):
        ref = None
        # transcripts sit beside the audio with the same stem
        for ext in (".txt", ".json"):
            cand = wav.with_suffix(ext)
            if cand.exists():
                if ext == ".json":
                    try:
                        d = json.loads(cand.read_text(encoding="utf-8"))
                        ref = " ".join(u.get("text", "") for u in d) if isinstance(d, list) else d.get("text")
                    except Exception:
                        ref = None
                else:
                    ref = cand.read_text(encoding="utf-8")
                if ref:
                    break
        if ref and ref.strip():
            pairs.append((wav, re.sub(r"\s+", " ", ref).strip()))
        if limit and len(pairs) >= limit:
            break
    return pairs


def score(refs, hyps):
    r = [NORM(x) for x in refs]
    h = [NORM(x) for x in hyps]
    keep = [(a, b) for a, b in zip(r, h) if a.strip()]
    if not keep:
        return 100.0, 100.0
    r, h = zip(*keep)
    return 100 * jiwer.wer(list(r), list(h)), 100 * jiwer.cer(list(r), list(h))


def run_whisper(name, pairs, device):
    from transcribe import load_model, transcribe
    model = load_model(name, device)
    hyps, audio_s, proc_s = [], 0.0, 0.0
    for i, (wav, _) in enumerate(pairs, 1):
        text, _, took, dur = transcribe(model, str(wav))
        hyps.append(text); audio_s += dur; proc_s += took
        print(f"    {i}/{len(pairs)}  {dur:5.1f}s audio -> {took:5.1f}s", flush=True)
    return hyps, audio_s, proc_s


def run_parakeet(pairs):
    """NVIDIA Parakeet TDT - benchmarks show ~6.3% WER and much faster on CPU."""
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    hyps, audio_s, proc_s = [], 0.0, 0.0
    import librosa
    for i, (wav, _) in enumerate(pairs, 1):
        dur = librosa.get_duration(path=str(wav))
        t0 = time.time()
        out = m.transcribe([str(wav)])
        took = time.time() - t0
        hyps.append(out[0].text if hasattr(out[0], "text") else str(out[0]))
        audio_s += dur; proc_s += took
        print(f"    {i}/{len(pairs)}  {dur:5.1f}s audio -> {took:5.1f}s", flush=True)
    return hyps, audio_s, proc_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit consultations")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", default="small.en,distil-large-v3")
    ap.add_argument("--with-parakeet", action="store_true")
    a = ap.parse_args()

    if not REPO.exists():
        raise SystemExit("PriMock57 not found. Run:  python get_primock57.py")

    pairs = load_pairs(a.n)
    if not pairs:
        raise SystemExit("No (audio, transcript) pairs found - is git-lfs pulled?")
    print(f"benchmarking on {len(pairs)} consultations\n")

    results = []
    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        print(f"--- {name} ---")
        hyps, audio_s, proc_s = run_whisper(name, pairs, a.device)
        wer, cer = score([r for _, r in pairs], hyps)
        results.append((name, wer, cer, audio_s / proc_s))

    if a.with_parakeet:
        print("--- parakeet-tdt-0.6b ---")
        try:
            hyps, audio_s, proc_s = run_parakeet(pairs)
            wer, cer = score([r for _, r in pairs], hyps)
            results.append(("parakeet-tdt-0.6b", wer, cer, audio_s / proc_s))
        except ImportError:
            print("    skipped - install with:  pip install nemo_toolkit[asr]")

    print("\n" + "=" * 72)
    print(f"  RESULTS on PriMock57 ({len(pairs)} real consultations, {a.device})")
    print("=" * 72)
    print(f"  {'model':<24} {'WER':>7} {'word acc':>9} {'CER':>7} {'char acc':>9} {'speed':>9}")
    print("  " + "-" * 68)
    for n, w, c, x in sorted(results, key=lambda r: r[1]):
        print(f"  {n:<24} {w:6.2f}% {100-w:8.1f}% {c:6.2f}% {100-c:8.1f}% {x:7.1f}x RT")
    print("=" * 72)
    print("  WER/CER are ERROR rates - LOWER IS BETTER. speed = x faster than real-time.")

    (HERE / "benchmark_results.json").write_text(json.dumps(
        [{"model": n, "wer": w, "cer": c, "speed_x_realtime": x} for n, w, c, x in results],
        indent=2), encoding="utf-8")
    print(f"\n  saved -> benchmark_results.json")


if __name__ == "__main__":
    main()
