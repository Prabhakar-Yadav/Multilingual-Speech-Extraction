"""Measure BOTH word accuracy (WER) and character accuracy (CER) of the dialect
model on clean held-out Vaani. For Bengali, character accuracy is the fair metric."""
import pickle, torch, jiwer
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
rows = pickle.load(open(B / "data" / "bn_clips_dialect.pkl", "rb"))[:60]

model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=DT).to(DEV)
model = PeftModel.from_pretrained(model, ADAPTER)
model.generation_config.language = "bengali"; model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.eval()

refs, hyps = [], []
for r in rows:
    feat = proc.feature_extractor(r["wav"], sampling_rate=16000, return_tensors="pt").input_features.to(DEV).to(DT)
    with torch.no_grad():
        ids = model.generate(feat, max_new_tokens=225)
    hyps.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    refs.append(r["text"])

wer = 100 * jiwer.wer(refs, hyps)
cer = 100 * jiwer.cer(refs, hyps)
print(f"\n===== DIALECT model on clean Vaani ({len(rows)} held-out clips) =====")
print(f"  Word Error Rate (WER):      {wer:5.1f}%   ->  word accuracy      {100-wer:5.1f}%")
print(f"  Character Error Rate (CER): {cer:5.1f}%   ->  CHARACTER accuracy {100-cer:5.1f}%")
