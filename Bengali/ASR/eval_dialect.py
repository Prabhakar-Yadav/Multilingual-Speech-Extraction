"""AFTER TRAINING: transcribe the 4 client calls with the DIALECT-tuned model and
score WER against the client's ground truth (test-only). Compares to the 94% baseline
from the Kolkata-trained model.

    ..\\.venv-train\\Scripts\\python.exe -u Bengali\\ASR\\eval_dialect.py
"""
import json, pickle, torch, numpy as np, jiwer
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

B = Path(__file__).resolve().parent
VS = B.parents[1]
BASE = str(B / "models" / "whisper-medium")
ADAPTER = str(B / "models" / "whisper-bn-dialect-lora")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
CHUNK = 30 * 16000

torch.cuda.set_per_process_memory_fraction(0.87) if DEV == "cuda" else None
proc = WhisperProcessor.from_pretrained(BASE, language="bengali", task="transcribe")
wavs = pickle.load(open(B / "data" / "four_wavs.pkl", "rb"))
gold = json.loads((VS / "data" / "gold.json").read_text(encoding="utf-8"))

def collapse(t):
    out = []
    for w in t.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=DT).to(DEV)
model = PeftModel.from_pretrained(model, ADAPTER)
model.generation_config.language = "bengali"; model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.eval()
print("dialect model loaded", flush=True)

wers = []
for i, wav in enumerate(wavs, 1):
    parts = []
    for s in range(0, len(wav), CHUNK):
        c = wav[s:s + CHUNK]
        if len(c) < 16000:
            break
        feat = proc.feature_extractor(c, sampling_rate=16000, return_tensors="pt").input_features.to(DEV).to(DT)
        with torch.no_grad():
            ids = model.generate(feat, max_new_tokens=225)
        parts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    hyp = collapse(" ".join(parts))
    best = min(gold, key=lambda g: jiwer.wer(g["bengali"], hyp))
    w = 100 * jiwer.wer(best["bengali"], hyp)
    wers.append(w)
    print(f"  file {i} -> Call {best['call']} ({best['patient']}): WER {w:.1f}%", flush=True)

avg = sum(wers) / len(wers)
print(f"\n===== DIALECT model avg WER: {avg:.1f}% =====")
print(f"      (Kolkata-trained baseline was 94.0% | base whisper 100%)")
print("      LOWER = BETTER. If this beats 94%, dialect matching helped.")
