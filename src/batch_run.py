"""Batch: every audio in data/ -> Bengali ASR -> translate -> extract terms.
RAM-safe: loads one model at a time. Writes outputs/result_<name>.json."""
import json, re, glob, gc
from pathlib import Path
import numpy as np
from faster_whisper.audio import decode_audio
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

B = Path(__file__).resolve().parent.parent; SR = 16000
EXT = (".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac", ".mpeg", ".aac")
CAT = {"Medication": "medicines", "Therapeutic_procedure": "procedures",
       "Diagnostic_procedure": "tests", "Lab_value": "tests",
       "Disease_disorder": "diagnoses", "Sign_symptom": "symptoms"}
files = sorted(f for f in glob.glob(str(B / "data" / "*")) if f.lower().endswith(EXT))
print("FILES", [Path(f).name for f in files], flush=True)

def clean(p):
    x = np.asarray(decode_audio(p, sampling_rate=SR), dtype=np.float32)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); X[f < 250] = 0
    y = np.fft.irfft(X, n=len(x)).astype(np.float32)
    return (y / (np.max(np.abs(y)) + 1e-9)) * 0.95

def deloop(t):
    t = re.sub(r"(.)\1{3,}", r"\1\1", t); t = re.sub(r"(\S+?)(?:[-\s]\1){2,}", r"\1", t)
    r = []
    for w in t.split():
        if not (r and r[-1] == w): r.append(w)
    return " ".join(r)

res = {Path(f).name: {} for f in files}

asr = pipeline("automatic-speech-recognition", model=str(B / "models" / "banglaasr"),
               chunk_length_s=30, stride_length_s=5, device="cpu")
for f in files:
    t = asr({"raw": clean(f), "sampling_rate": SR},
            generate_kwargs={"language": "bn", "task": "transcribe"}, batch_size=1)["text"]
    res[Path(f).name]["bn"] = deloop(t); print("ASR done", Path(f).name, flush=True)
del asr; gc.collect()

tok = AutoTokenizer.from_pretrained(str(B / "models" / "nllb-200-distilled-600M"), src_lang="ben_Beng")
mdl = AutoModelForSeq2SeqLM.from_pretrained(str(B / "models" / "nllb-200-distilled-600M")); mdl.eval()
bos = tok.convert_tokens_to_ids("eng_Latn")
def tr(t):
    s = [p.strip() for p in re.split(r"(?<=[।.?!])\s+", t.strip()) if p.strip()]
    o = []
    for i in range(0, len(s), 8):
        ip = tok(s[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            g = mdl.generate(**ip, forced_bos_token_id=bos, max_length=256, num_beams=5)
        o += tok.batch_decode(g, skip_special_tokens=True)
    return " ".join(o)
for f in files:
    res[Path(f).name]["en"] = tr(res[Path(f).name]["bn"])
del mdl, tok; gc.collect()

ner = pipeline("token-classification", model=str(B / "models" / "biomedical-ner-all"),
               aggregation_strategy="simple")
def ex(t):
    sp = sorted(ner(t), key=lambda s: s["start"]); m = []
    for s in sp:
        g = s["entity_group"]
        if m and m[-1]["g"] == g and s["start"] - m[-1]["e"] <= 1: m[-1]["e"] = s["end"]
        else: m.append({"g": g, "s": s["start"], "e": s["end"]})
    b = {v: [] for v in set(CAT.values())}
    for x in m:
        bk = CAT.get(x["g"])
        if bk:
            w = t[x["s"]:x["e"]].strip()
            if w and w.lower() not in {z.lower() for z in b[bk]}: b[bk].append(w)
    return b
for f in files:
    n = Path(f).name; res[n]["terms"] = ex(res[n]["en"])
    Path(B / "outputs" / f"result_{n}.json").write_text(
        json.dumps(res[n], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {n} ===\nEN: {res[n]['en'][:250]}\nTERMS: {json.dumps(res[n]['terms'], ensure_ascii=False)}", flush=True)
print("ALL_DONE", flush=True)
