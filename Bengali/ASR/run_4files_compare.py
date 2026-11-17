"""Transcribe the 4 real client files (FULL length, 30s-chunked) with BASE
whisper-medium vs our FINE-TUNED model. Auto-uses GPU if available, else CPU.
Run prep_4files.py first (decodes the mp3s torch-free -> four_wavs.pkl).

    .\.venv-train\Scripts\python.exe -u Bengali_ASR\run_4files_compare.py
"""
import json, pickle, torch, numpy as np                 # no librosa (segfaults with torch)
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

B = Path(__file__).resolve().parent
BASE = str(B / "models" / "whisper-medium")
ADAPTER = str(B / "models" / "whisper-bn-vaani-lora")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
CHUNK = 30 * 16000                                       # whisper processes 30s at a time

proc = WhisperProcessor.from_pretrained(BASE, language="bengali", task="transcribe")
wavs = pickle.load(open(B / "data" / "four_wavs.pkl", "rb"))
print(f"device={DEV}", flush=True)

def transcribe_full(model, wav):
    parts = []
    for s in range(0, len(wav), CHUNK):
        chunk = wav[s:s + CHUNK]
        if len(chunk) < 16000:                          # skip <1s tail
            break
        feat = proc.feature_extractor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(DEV).to(DT)
        with torch.no_grad():
            ids = model.generate(feat, max_new_tokens=225)
        parts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(parts)

def load(adapter=None):
    m = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=DT).to(DEV)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    m.generation_config.language = "bengali"; m.generation_config.task = "transcribe"
    m.generation_config.forced_decoder_ids = None
    return m

print("\n=== BASE whisper-medium ===", flush=True)
base = load()
base_out = [transcribe_full(base, w) for w in wavs]
for i, t in enumerate(base_out, 1): print(f"File {i}: {t}\n", flush=True)
del base
if DEV == "cuda": torch.cuda.empty_cache()

print("=== FINE-TUNED (Vaani degraded-audio) ===", flush=True)
ft = load(ADAPTER)
ft_out = [transcribe_full(ft, w) for w in wavs]
for i, t in enumerate(ft_out, 1): print(f"File {i}: {t}\n", flush=True)

(B / "four_files_compare.json").write_text(
    json.dumps({"base": base_out, "finetuned": ft_out}, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved -> Bengali_ASR/four_files_compare.json", flush=True)
