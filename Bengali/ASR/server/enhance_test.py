"""
Does audio enhancement actually help OUR client calls?

The literature splits two ways, so we test rather than assume:
  * bandwidth extension / super-resolution   -> reported to help a lot on 8kHz
  * denoising front-ends                     -> repeatedly reported to HURT
    modern ASR (artifacts the acoustic model was never trained on)

Our audio is 8 kHz with 90% of its energy under ~1 kHz, so this is exactly the
regime where those two findings collide. Each variant is scored with the same
fine-tuned model and the same gold matching as eval_client.py.

    python enhance_test.py                 # all variants, 2 files
    FILES=4 python enhance_test.py         # all 4 calls
"""
import json, os, time
from pathlib import Path

import jiwer, librosa, numpy as np, soundfile as sf, torch
from peft import PeftModel
from scipy import signal
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          pipeline)

HOME  = Path.home() / "bengali_asr"
AUDIO = HOME / "client_audio"
WORK  = HOME / "enh_work"; WORK.mkdir(exist_ok=True)
GOLD  = json.loads((AUDIO / "gold.json").read_text(encoding="utf-8"))
CKPT  = HOME / "ckpt_run2" / "checkpoint-3000"
NFILES = int(os.environ.get("FILES", 2))
SR = 16000   # what Whisper expects


# ---------------- enhancement variants ----------------
def v_raw(y, sr):
    """Plain resample to 16k - what the model already sees. The control."""
    return y


def v_norm(y, sr):
    """Peak-normalise + gentle pre-emphasis. Cheapest possible fix: the calls sit
    at rms~0.045 (quiet), and pre-emphasis lifts the consonant range."""
    y = signal.lfilter([1, -0.97], [1], y)
    return y / (np.abs(y).max() + 1e-9) * 0.95


def v_bandlift(y, sr):
    """Crude bandwidth extension: spectral folding. Copies the 1-4kHz band up to
    4-8kHz at reduced gain, so the empty upper half of the spectrum carries
    something correlated with the speech instead of pure silence."""
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / sr)
    out = Y.copy()
    src = (f >= 1000) & (f < 4000)
    dst_start = np.searchsorted(f, 4000)
    n = min(src.sum(), len(f) - dst_start)
    out[dst_start:dst_start + n] += Y[src][:n] * 0.25
    y2 = np.fft.irfft(out, n=len(y))
    return y2 / (np.abs(y2).max() + 1e-9) * 0.95


def v_denoise(y, sr):
    """Spectral-gating denoise. Literature says this often HURTS modern ASR -
    we test it precisely because it is the 'obvious' thing to try."""
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.7)
    except ImportError:
        return None


VARIANTS = [("raw", v_raw), ("norm+preemph", v_norm),
            ("bandlift", v_bandlift), ("denoise", v_denoise)]

# ---------------- model ----------------
print(f"loading model + {CKPT.name}", flush=True)
proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
m = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3", dtype=torch.float16).to("cuda")
m = PeftModel.from_pretrained(m, str(CKPT)).merge_and_unload().eval()
asr = pipeline("automatic-speech-recognition", model=m, tokenizer=proc.tokenizer,
               feature_extractor=proc.feature_extractor, chunk_length_s=30,
               stride_length_s=5, torch_dtype=torch.float16, device="cuda")


def collapse(t):
    out = []
    for w in t.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)


files = sorted(AUDIO.glob("*.mpeg"))[:NFILES]
results = {}

for vname, fn in VARIANTS:
    wers, cers = [], []
    for f in files:
        y, _ = librosa.load(str(f), sr=SR)      # resamples 8k -> 16k
        y2 = fn(y, SR)
        if y2 is None:
            print(f"  {vname}: skipped (missing dependency)"); wers = None; break
        p = WORK / f"{f.stem}_{vname}.wav"
        sf.write(p, y2.astype(np.float32), SR)

        t0 = time.time()
        hyp = collapse(asr(str(p), generate_kwargs={"language": "bengali",
                                                    "task": "transcribe"})["text"].strip())
        best = min(GOLD, key=lambda g: jiwer.wer(g["bengali"], hyp))
        w = 100 * jiwer.wer(best["bengali"], hyp)
        c = 100 * jiwer.cer(best["bengali"], hyp)
        wers.append(w); cers.append(c)
        print(f"  {vname:14} {f.name:14} WER {w:5.1f}%  CER {c:5.1f}%  [{time.time()-t0:.0f}s]",
              flush=True)
    if wers:
        results[vname] = (sum(wers) / len(wers), sum(cers) / len(cers))

print("\n" + "=" * 60)
print(f"  ENHANCEMENT COMPARISON ({len(files)} client calls)")
print("=" * 60)
base = results.get("raw", (None, None))[1]
for k, (w, c) in sorted(results.items(), key=lambda kv: kv[1][1]):
    delta = "" if base is None else f"  (CER {c-base:+.1f} vs raw)"
    print(f"  {k:14} WER {w:5.1f}%   CER {c:5.1f}%{delta}")
print("=" * 60)
print("  LOWER IS BETTER. CER is the metric that matters here - at ~90% WER,")
print("  WER is saturated and can't show small real improvements.")
