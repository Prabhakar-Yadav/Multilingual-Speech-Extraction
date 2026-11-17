"""ASR bake-off: run every Bengali ASR model we have on the 4 real client files and
measure WER against the client's ground-truth Bengali. Finds which model handles the
degraded phone audio best. Run AFTER the GPU is free.

    ..\..\..\.venv-train\Scripts\python.exe -u Bengali\ASR\eval_asr_models.py
"""
import json, pickle, torch, numpy as np
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import jiwer

ASR = Path(__file__).resolve().parent
VS = ASR.parents[1]                                   # Voice-AI
MODELS = VS / "models"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 30 * 16000

gold = json.loads((VS / "data" / "gold.json").read_text(encoding="utf-8"))
wavs = pickle.load(open(ASR / "data" / "four_wavs.pkl", "rb"))

def collapse(t):
    out = []
    for w in t.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

def transcribe(model, proc, wav, dt):
    parts = []
    for s in range(0, len(wav), CHUNK):
        c = wav[s:s + CHUNK]
        if len(c) < 16000:
            break
        feat = proc.feature_extractor(c, sampling_rate=16000, return_tensors="pt").input_features.to(DEV).to(dt)
        with torch.no_grad():
            try:
                ids = model.generate(feat, max_new_tokens=200, language="bengali", task="transcribe")
            except Exception:
                ids = model.generate(feat, max_new_tokens=200)
        parts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(parts)

def run(name, load):
    try:
        model, proc, dt = load()
        model.eval()
        wers = []
        for wav in wavs:
            hyp = collapse(transcribe(model, proc, wav, dt))
            best = min(gold, key=lambda g: jiwer.wer(g["bengali"], hyp))
            wers.append(100 * jiwer.wer(best["bengali"], hyp))
        print(f"{name:35}: avg WER {sum(wers)/len(wers):5.1f}%   per-file {[round(w) for w in wers]}", flush=True)
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"{name:35}: FAILED - {str(e)[:90]}", flush=True)

def whisper_load(path, dtype):
    proc = WhisperProcessor.from_pretrained(str(path))
    m = WhisperForConditionalGeneration.from_pretrained(str(path), torch_dtype=dtype).to(DEV)
    return m, proc, dtype

DT = torch.float16 if DEV == "cuda" else torch.float32
print(f"device={DEV}\n", flush=True)
run("banglaasr (whisper-small)", lambda: whisper_load(MODELS / "banglaasr", DT))
run("vaani-medium (whisper-medium)", lambda: whisper_load(MODELS / "vaani-medium", DT))
run("medibeng (medical whisper)", lambda: whisper_load(MODELS / "medibeng", DT))

def ours():
    proc = WhisperProcessor.from_pretrained(str(ASR / "models" / "whisper-medium"))
    m = WhisperForConditionalGeneration.from_pretrained(str(ASR / "models" / "whisper-medium"), torch_dtype=torch.bfloat16).to(DEV)
    m = PeftModel.from_pretrained(m, str(ASR / "models" / "whisper-bn-vaani-lora"))
    return m, proc, torch.bfloat16
run("OUR fine-tuned (vaani-lora)", ours)
print("\n(lower WER = better; base whisper-medium was ~100%)", flush=True)
