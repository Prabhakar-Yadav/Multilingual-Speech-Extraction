"""HONEST held-out eval: reproduces EXACTLY the trainer's split (same file, same
token filter, same seed-0 shuffle, same first-N as eval) so we never test on clips
the model trained on. Reports WER (word) and CER (character) accuracy."""
import pickle, random, torch, jiwer
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

B = Path(__file__).resolve().parent
BASE = str(B / "models" / "whisper-medium")
ADAPTER = str(B / "models" / "whisper-bn-vaani-max")
N_EVAL = 80
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.87)

processor = WhisperProcessor.from_pretrained(BASE, language="bengali", task="transcribe")

# --- reproduce the trainer's split EXACTLY ---
rows = pickle.load(open(B / "data" / "bn_clips_all.pkl", "rb"))
rows = [r for r in rows if len(processor.tokenizer(r["text"]).input_ids) <= 440]
random.seed(0); random.shuffle(rows)
eval_rows = rows[:N_EVAL]                       # <- the clips the model NEVER trained on
print(f"held-out clips: {len(eval_rows)} (never seen in training)", flush=True)

model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=DT).to(DEV)
model = PeftModel.from_pretrained(model, ADAPTER)
model.generation_config.language = "bengali"; model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.eval()

refs, hyps = [], []
for r in eval_rows:
    feat = processor.feature_extractor(r["wav"], sampling_rate=16000, return_tensors="pt").input_features.to(DEV).to(DT)
    with torch.no_grad():
        ids = model.generate(feat, max_new_tokens=225)
    hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    refs.append(r["text"])

wer = 100 * jiwer.wer(refs, hyps)
cer = 100 * jiwer.cer(refs, hyps)
print(f"\n===== HONEST held-out clean Vaani ({len(refs)} unseen clips) =====")
print(f"  WER {wer:5.1f}%  ->  word accuracy      {100-wer:5.1f}%")
print(f"  CER {cer:5.1f}%  ->  CHARACTER accuracy {100-cer:5.1f}%")
