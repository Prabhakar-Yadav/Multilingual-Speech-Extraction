"""Measure the dialect model's WER on CLEAN Vaani (no degradation) - the honest
in-domain quality number. The training eval degraded the audio, which is why it
looked like 56%; this shows the real quality on clean speech."""
import pickle, torch, numpy as np, jiwer
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

B = Path(__file__).resolve().parent
BASE = str(B / "models" / "whisper-medium")
ADAPTER = str(B / "models" / "whisper-bn-dialect-lora")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.87)

proc = WhisperProcessor.from_pretrained(BASE, language="bengali", task="transcribe")
rows = pickle.load(open(B / "data" / "bn_clips_dialect.pkl", "rb"))[:40]   # same held-out eval clips

def transcribe(model, wav):
    feat = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(DEV).to(DT)
    with torch.no_grad():
        ids = model.generate(feat, max_new_tokens=225)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

refs = [r["text"] for r in rows]

# base whisper-medium (no fine-tune) on clean audio
base = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=DT).to(DEV)
base.generation_config.language = "bengali"; base.generation_config.task = "transcribe"; base.generation_config.forced_decoder_ids = None
base_h = [transcribe(base, r["wav"]) for r in rows]
print(f"BASE whisper-medium, CLEAN Vaani WER: {100*jiwer.wer(refs, base_h):.1f}%", flush=True)
del base
if DEV == "cuda": torch.cuda.empty_cache()

# dialect fine-tuned on clean audio
ft = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=DT).to(DEV)
ft = PeftModel.from_pretrained(ft, ADAPTER)
ft.generation_config.language = "bengali"; ft.generation_config.task = "transcribe"; ft.generation_config.forced_decoder_ids = None
ft_h = [transcribe(ft, r["wav"]) for r in rows]
print(f"DIALECT fine-tuned, CLEAN Vaani WER: {100*jiwer.wer(refs, ft_h):.1f}%", flush=True)
print("(the 56% we saw was on DEGRADED audio; this is the clean, honest number)", flush=True)
