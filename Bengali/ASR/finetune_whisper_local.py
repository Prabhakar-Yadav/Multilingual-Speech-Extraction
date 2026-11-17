"""
STEP 2: BENGALI ASR FINE-TUNE - LOCAL (RTX 3050 4GB), whisper-medium 8-bit.
Loads the pickle made by prep_data.py (NO pyarrow here - it segfaults with torch).
Degrades audio to match the client's phone recordings, fine-tunes, reports WER.

    .\.venv-train\Scripts\python.exe -u Bengali_ASR\finetune_whisper_local.py

Keep the laptop plugged in (sleep disabled). If it OOMs, lower to whisper-small.
"""
import random, pickle, numpy as np, torch
from pathlib import Path
from math import gcd
from scipy.signal import butter, lfilter, resample_poly
import jiwer
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, BitsAndBytesConfig,
                          EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass

B = Path(__file__).resolve().parent
BASE = str(B / "models" / "whisper-medium")
N_EVAL = 80           # bigger held-out test for a reliable WER/CER number

# SAFETY: the NVIDIA driver BSOD'd twice when PyTorch ran the GPU to 97% VRAM.
# Hard-cap the process -> clean CUDA-OOM instead of a driver crash / blue screen.
torch.cuda.set_per_process_memory_fraction(0.87)   # ~3.55GB PyTorch + ~0.25GB context = ~3.8GB total (user's limit)
processor = WhisperProcessor.from_pretrained(BASE, language="bengali", task="transcribe")

# ---- augmentation (scipy only): degrade to match the client's 4 files (<1kHz energy) ----
def lowpass(w, sr, c):
    b, a = butter(6, c / (sr / 2), btype="low"); return lfilter(b, a, w).astype(np.float32)
def rs(w, sr, tgt):
    g = gcd(sr, tgt); return resample_poly(w, tgt // g, sr // g).astype(np.float32)
def degrade(wav, sr=16000, p=0.8):
    if random.random() > p: return wav.astype(np.float32)
    w = lowpass(wav.astype(np.float32), sr, random.uniform(1200, 3400))
    tgt = random.choice([8000, 8000, 4000]); w = rs(w, sr, tgt); w = rs(w, tgt, sr)
    snr = random.uniform(3, 18); rms = np.sqrt(np.mean(w ** 2)) + 1e-9
    out = w + np.random.randn(len(w)).astype(np.float32) * rms / (10 ** (snr / 20))
    return np.nan_to_num(out).astype(np.float32)      # guard against any non-finite values

rows = pickle.load(open(B / "data" / "bn_clips_all.pkl", "rb"))   # ALL Vaani: dialect + Kolkata (3298 clips)
before = len(rows)                                    # whisper decoder max = 448 tokens
rows = [r for r in rows if len(processor.tokenizer(r["text"]).input_ids) <= 440]
random.seed(0); random.shuffle(rows)
print(f"loaded {len(rows)}/{before} clips within token limit ({len(rows)-N_EVAL} train / {N_EVAL} eval)", flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        # CLEAN audio (no degradation): the goal is lowest WER on clean Vaani
        return {"input_features": processor.feature_extractor(r["wav"], sampling_rate=16000).input_features[0],
                "labels": processor.tokenizer(r["text"]).input_ids}
eval_ds, train_ds = DS(rows[:N_EVAL]), DS(rows[N_EVAL:])

@dataclass
class Collator:
    processor: object
    def __call__(self, features):
        batch = self.processor.feature_extractor.pad([{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)   # match bf16 model (fixes eval/generate)
        labs = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        batch["labels"] = labs["input_ids"].masked_fill(labs.attention_mask.ne(1), -100)
        return batch

def compute_metrics(pred):
    pi, li = pred.predictions, pred.label_ids
    li = np.where(li == -100, processor.tokenizer.pad_token_id, li)
    ps = processor.batch_decode(pi, skip_special_tokens=True); ls = processor.batch_decode(li, skip_special_tokens=True)
    pairs = [(r, p) for r, p in zip(ls, ps) if r.strip()]
    if not pairs: return {"wer": 100.0}
    r, p = zip(*pairs); return {"wer": 100 * jiwer.wer(list(r), list(p))}

model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16)
model.to("cuda")
model.generation_config.language = "bengali"; model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))   # more layers = more capacity
model.print_trainable_parameters()

args = Seq2SeqTrainingArguments(
    output_dir=str(B / "outputs_max"), per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=5e-4, warmup_steps=40, num_train_epochs=3, bf16=True, max_grad_norm=1.0,   # 3298 clips x 3 epochs (~10k examples)
    optim="paged_adamw_8bit", gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # checkpoint every 50 steps (~25 min) so a crash/reboot loses almost nothing.
    # (steps-based saving is incompatible with load_best_model_at_end, so we keep the
    #  per-epoch WER curve in the log and pick the best checkpoint ourselves afterwards.)
    per_device_eval_batch_size=1, eval_strategy="epoch", save_strategy="steps", save_steps=100,
    save_total_limit=2,
    predict_with_generate=True, generation_max_length=225, logging_steps=20,
    report_to="none", remove_unused_columns=False)
import glob as _glob                      # auto-resume: an 8h run WILL get interrupted
_cks = [c for c in _glob.glob(str(B / "outputs_max" / "checkpoint-*"))
        if (Path(c) / "trainer_state.json").exists()]     # skip half-written checkpoints
_cks.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
_resume = _cks[-1] if _cks else None
print("resuming from:", _resume, flush=True)

Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
    data_collator=Collator(processor), compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
    ).train(resume_from_checkpoint=_resume)   # no early-stop: it needs load_best, which blocks step-saving

out = B / "models" / "whisper-bn-vaani-max"           # the max-accuracy clean-Vaani model
model.save_pretrained(str(out)); processor.save_pretrained(str(out))
print("saved ->", out, flush=True)
print("Compare epoch-1 WER vs final WER above (lower = better).", flush=True)
