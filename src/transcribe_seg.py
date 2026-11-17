"""Segment-wise decode with BanglaASR: energy-based VAD cuts audio into short
utterances; each is decoded ALONE (avoids transformers' buggy long-form chunking).
    python src/transcribe_seg.py <audio.wav> <out.json>
"""
import sys, json, time, wave
from pathlib import Path
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

B = Path(__file__).resolve().parent.parent
audio, out = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else str(B/"models"/"banglaasr")
LANG = sys.argv[4] if len(sys.argv) > 4 else "bn"

with wave.open(audio, "rb") as w:
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

# energy VAD: 30ms frames, split on >=400ms silence, segments 1-28s
fl = int(0.03 * sr)
fr = x[: len(x)//fl*fl].reshape(-1, fl)
e = np.sqrt((fr**2).mean(1))
act = e > max(e.mean()*0.35, 1e-4)
segs, s = [], None
for i, a in enumerate(act):
    if a and s is None: s = i
    if not a and s is not None and (i - np.argmax(act[s:i][::-1])) and all(~act[i:i+14]):
        segs.append((s, i)); s = None
if s is not None: segs.append((s, len(act)))
# merge tiny gaps, enforce max 28s
merged = []
for a, b in segs:
    if merged and (a - merged[-1][1]) * 0.03 < 0.4: merged[-1] = (merged[-1][0], b)
    else: merged.append((a, b))
chunks = []
for a, b in merged:
    if (b - a) * 0.03 < 0.6: continue
    while (b - a) * 0.03 > 28:
        chunks.append((a, a + int(28/0.03))); a += int(28/0.03)
    chunks.append((a, b))

proc = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL); model.eval()
# MediBeng was trained on task=translate (bn speech -> EN text)
task = "translate" if "medibeng" in MODEL.lower() else "transcribe"

parts = []
t0 = time.time()
for a, b in chunks:
    seg = x[a*fl:b*fl]
    feats = proc(seg, sampling_rate=sr, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feats, language=LANG, task=task, max_new_tokens=200,
                             num_beams=4, no_repeat_ngram_size=4)
    txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if txt:
        parts.append(txt)
        print(f"[{a*0.03:6.1f}s] {txt}", flush=True)

res = {"detected_language": "bn", "model": "banglaasr-segmented",
       "processing_sec": round(time.time()-t0, 1), "transcript": " ".join(parts)}
Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("DONE", out, flush=True)
